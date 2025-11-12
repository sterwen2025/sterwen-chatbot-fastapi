import { useState, useEffect, useRef } from 'react';
import {
  Layout,
  Card,
  Input,
  Button,
  Checkbox,
  DatePicker,
  Select,
  Space,
  message,
  Spin,
  Typography,
  Divider,
  Tag,
  Empty,
  Row,
  Col
} from 'antd';
import {
  SendOutlined,
  ReloadOutlined,
  DeleteOutlined,
  ArrowLeftOutlined,
  MessageOutlined,
  FilterOutlined,
  DatabaseOutlined,
  RobotOutlined
} from '@ant-design/icons';
import { API_ENDPOINTS } from '../config/api';
import dayjs, { Dayjs } from 'dayjs';
import ReactMarkdown from 'react-markdown';

const { Header, Sider, Content } = Layout;
const { TextArea } = Input;
const { Text, Title } = Typography;
const { RangePicker } = DatePicker;

interface ChatMessage {
  question: string;
  answer: string;
  sources: string[];
  timestamp: Date;
}

interface FilterStats {
  meeting_notes_count: number;
  factsheet_comments_count: number;
  transcripts_count: number;
  total_count: number;
}

const ChatBot = () => {
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // State management
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(false);

  // Filter states
  // Note: Only Meeting Notes has RAG implementation. Factsheet Comments and Transcripts are disabled for now.
  const [dataSources, setDataSources] = useState<string[]>(['Meeting Notes']);
  const [useDateFilter, setUseDateFilter] = useState(false);
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs]>([dayjs().subtract(6, 'month'), dayjs()]);
  const [selectedPortfolios, setSelectedPortfolios] = useState<string[]>([]);
  const [selectedFunds, setSelectedFunds] = useState<string[]>([]);

  // Data lists
  const [allFunds, setAllFunds] = useState<string[]>([]);
  const [allPortfolios, setAllPortfolios] = useState<string[]>([]);
  const [filterStats, setFilterStats] = useState<FilterStats>({
    meeting_notes_count: 0,
    factsheet_comments_count: 0,
    transcripts_count: 0,
    total_count: 0
  });

  // Fetch funds and portfolios on mount
  useEffect(() => {
    fetchFunds();
    fetchPortfolios();
  }, []);

  // Fetch stats when filters change
  useEffect(() => {
    fetchStats();
  }, [dataSources, useDateFilter, dateRange, selectedFunds]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  const fetchFunds = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.chatFunds);
      const data = await response.json();
      setAllFunds(data.funds);
    } catch (error) {
      console.error('Error fetching funds:', error);
    }
  };

  const fetchPortfolios = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.chatPortfolios);
      const data = await response.json();
      setAllPortfolios(data.portfolios);
    } catch (error) {
      console.error('Error fetching portfolios:', error);
    }
  };

  const fetchFundsByPortfolios = async (portfolios: string[]) => {
    if (!portfolios || portfolios.length === 0) return;

    try {
      const response = await fetch(API_ENDPOINTS.chatPortfoliosFunds, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ portfolios })
      });
      const data = await response.json();
      setSelectedFunds(data.funds);
    } catch (error) {
      console.error('Error fetching funds by portfolios:', error);
    }
  };

  const fetchStats = async () => {
    if (dataSources.length === 0) return;

    setStatsLoading(true);
    try {
      const response = await fetch(API_ENDPOINTS.chatStats, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: '',
          data_sources: dataSources,
          start_date: useDateFilter ? dateRange[0].format('YYYY-MM-DD') : null,
          end_date: useDateFilter ? dateRange[1].format('YYYY-MM-DD') : null,
          selected_funds: selectedFunds.length > 0 ? selectedFunds : null
        })
      });
      const data = await response.json();
      setFilterStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setStatsLoading(false);
    }
  };

  const handlePortfolioChange = (portfolios: string[]) => {
    setSelectedPortfolios(portfolios);
    if (portfolios.length > 0) {
      fetchFundsByPortfolios(portfolios);
    } else {
      setSelectedFunds([]);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      message.warning('Please enter a question');
      return;
    }

    if (dataSources.length === 0) {
      message.warning('Please select at least one data source');
      return;
    }

    setLoading(true);
    const currentQuestion = question;
    setQuestion(''); // Clear input immediately

    // Create a temporary message with empty answer for streaming
    const tempMessage: ChatMessage = {
      question: currentQuestion,
      answer: '',
      sources: dataSources,
      timestamp: new Date()
    };

    setConversation(prev => [...prev, tempMessage]);

    try {
      const response = await fetch(API_ENDPOINTS.chatAskStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: currentQuestion,
          data_sources: dataSources,
          start_date: useDateFilter ? dateRange[0].format('YYYY-MM-DD') : null,
          end_date: useDateFilter ? dateRange[1].format('YYYY-MM-DD') : null,
          selected_funds: selectedFunds.length > 0 ? selectedFunds : null,
          conversation_history: conversation.map(msg => ({
            question: msg.question,
            answer: msg.answer
          }))
        })
      });

      if (!response.ok || !response.body) {
        throw new Error('Stream response not available');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedAnswer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonData = JSON.parse(line.slice(6));

              if (jsonData.content) {
                accumulatedAnswer += jsonData.content;
                // Update the last message in conversation with accumulated answer
                setConversation(prev => {
                  const newConv = [...prev];
                  newConv[newConv.length - 1] = {
                    ...newConv[newConv.length - 1],
                    answer: accumulatedAnswer
                  };
                  return newConv;
                });
              }

              if (jsonData.done) {
                message.success('Answer received!');
              }

              if (jsonData.error) {
                throw new Error(jsonData.error);
              }
            } catch (e) {
              // Ignore JSON parse errors for partial chunks
              if (e instanceof Error && !e.message.includes('Unexpected')) {
                throw e;
              }
            }
          }
        }
      }
    } catch (error) {
      message.error('Failed to get answer. Please try again.');
      console.error('Error asking question:', error);
      // Remove the temporary message if there was an error
      setConversation(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const handleClearConversation = () => {
    setConversation([]);
    message.success('Conversation cleared');
  };

  return (
    <Layout style={{ height: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <Header style={{
        background: 'linear-gradient(to bottom, #ffffff 0%, #fafbfc 100%)',
        borderBottom: '1px solid #e1e4e8',
        padding: '0 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        zIndex: 1000,
        height: 70,
        lineHeight: '70px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.04)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img
            src="/sterwen-logo.png"
            alt="Sterwen"
            style={{ height: '42px', width: 'auto' }}
          />
        </div>
        <Space size={12}>
          <Button
            icon={<DeleteOutlined />}
            onClick={handleClearConversation}
            disabled={conversation.length === 0}
            type="text"
            danger={conversation.length > 0}
            style={{
              borderRadius: 8,
              height: 38,
              padding: '0 16px',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
          >
            Clear Chat
          </Button>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => window.location.href = 'https://sterwen-dashboard-cyc3embxhrhwhdbg.switzerlandnorth-01.azurewebsites.net'}
            type="default"
            style={{
              borderRadius: 8,
              height: 38,
              padding: '0 16px',
              fontWeight: 500,
              borderColor: '#d1d5db',
              transition: 'all 0.2s'
            }}
          >
            Back to Dashboard
          </Button>
        </Space>
      </Header>

      <Layout style={{ height: 'calc(100vh - 70px)' }}>
        {/* Sidebar - Filters */}
        <Sider
          width={300}
          style={{
            background: '#f8f9fa',
            borderRight: '1px solid #e1e4e8',
            overflow: 'auto'
          }}
        >
          <div style={{ padding: '24px 20px' }}>
            <div style={{
              marginBottom: 20,
              paddingBottom: 16,
              borderBottom: '2px solid #e1e4e8'
            }}>
              <Text strong style={{
                fontSize: 15,
                color: '#374151',
                letterSpacing: '0.5px',
                textTransform: 'uppercase'
              }}>
                Filters
              </Text>
            </div>

            {/* Data Sources */}
            <div style={{ marginBottom: 18 }}>
              <Text style={{ fontSize: 15, color: '#666', display: 'block', marginBottom: 8 }}>
                Data Sources
              </Text>
              <Checkbox.Group
                style={{ display: 'flex', flexDirection: 'column' }}
                value={dataSources}
                onChange={(values) => setDataSources(values as string[])}
              >
                <Checkbox value="Meeting Notes" style={{ marginLeft: 0, marginBottom: 6, fontSize: 15 }}>
                  <Text style={{ fontSize: 15 }}>Meeting Notes</Text>
                </Checkbox>
                <Checkbox value="Factsheet Comments" style={{ marginLeft: 0, marginBottom: 6, fontSize: 15 }}>
                  <Text style={{ fontSize: 15 }}>Factsheet Comments</Text>
                </Checkbox>
                <Checkbox value="Transcripts" style={{ marginLeft: 0, fontSize: 15 }}>
                  <Text style={{ fontSize: 15 }}>Transcripts</Text>
                </Checkbox>
              </Checkbox.Group>
            </div>

            {/* Date Range */}
            <div style={{ marginBottom: 18 }}>
              <Checkbox
                checked={useDateFilter}
                onChange={(e) => setUseDateFilter(e.target.checked)}
                style={{ marginBottom: 8, fontSize: 15 }}
              >
                <Text style={{ fontSize: 15, color: '#666' }}>Filter by Date</Text>
              </Checkbox>
              {useDateFilter && (
                <RangePicker
                  value={dateRange}
                  onChange={(dates) => dates && setDateRange([dates[0]!, dates[1]!])}
                  style={{ width: '100%', fontSize: 15 }}
                  format="YYYY-MM-DD"
                  size="large"
                />
              )}
            </div>

            {/* Portfolios */}
            <div style={{ marginBottom: 18 }}>
              <Text style={{ fontSize: 15, color: '#666', display: 'block', marginBottom: 6 }}>
                Portfolios
              </Text>
              <Select
                mode="multiple"
                placeholder="Select"
                size="large"
                style={{ width: '100%', fontSize: 15 }}
                options={allPortfolios.map(p => ({ label: p, value: p }))}
                value={selectedPortfolios}
                onChange={handlePortfolioChange}
                maxTagCount={1}
              />
            </div>

            {/* Funds */}
            <div style={{ marginBottom: 18 }}>
              <Text style={{ fontSize: 15, color: '#666', display: 'block', marginBottom: 6 }}>
                Funds
              </Text>
              <Select
                mode="multiple"
                placeholder="Select"
                size="large"
                style={{ width: '100%', fontSize: 15 }}
                options={allFunds.map(f => ({ label: f, value: f }))}
                value={selectedFunds}
                onChange={setSelectedFunds}
                maxTagCount={1}
                showSearch
              />
            </div>

            {/* Live Stats */}
            <Divider style={{ margin: '16px 0 12px 0', borderColor: '#d9d9d9' }} />
            <div style={{ marginBottom: 16 }}>
              <Text style={{ fontSize: 15, color: '#666', display: 'block', marginBottom: 10 }}>
                Available Data
              </Text>
              <Spin spinning={statsLoading}>
                <Space direction="vertical" style={{ width: '100%' }} size={6}>
                  {dataSources.includes('Meeting Notes') && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={{ fontSize: 15, color: '#666' }}>Meeting Notes</Text>
                      <Text style={{ fontSize: 15, fontWeight: 500 }}>{filterStats.meeting_notes_count}</Text>
                    </div>
                  )}
                  {dataSources.includes('Factsheet Comments') && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={{ fontSize: 15, color: '#666' }}>Factsheet</Text>
                      <Text style={{ fontSize: 15, fontWeight: 500 }}>{filterStats.factsheet_comments_count}</Text>
                    </div>
                  )}
                  {dataSources.includes('Transcripts') && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={{ fontSize: 15, color: '#666' }}>Transcripts</Text>
                      <Text style={{ fontSize: 15, fontWeight: 500 }}>{filterStats.transcripts_count}</Text>
                    </div>
                  )}
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    paddingTop: 6,
                    marginTop: 6,
                    borderTop: '1px solid #e8e8e8'
                  }}>
                    <Text strong style={{ fontSize: 15 }}>Total</Text>
                    <Text strong style={{ fontSize: 15 }}>{filterStats.total_count}</Text>
                  </div>
                </Space>
              </Spin>
            </div>

            <Button
              icon={<ReloadOutlined />}
              onClick={() => { fetchFunds(); fetchPortfolios(); fetchStats(); }}
              style={{ width: '100%', fontSize: 15 }}
              size="large"
            >
              Refresh
            </Button>
          </div>
        </Sider>

        {/* Chat Area */}
        <Content style={{
          display: 'flex',
          flexDirection: 'column',
          background: '#f5f5f5',
          height: '100%'
        }}>
          {/* Messages Area */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '24px',
            display: 'flex',
            flexDirection: 'column'
          }}>
            {conversation.length === 0 ? (
              <div style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Space direction="vertical" align="center" size={24}>
                  <div style={{
                    width: 80,
                    height: 80,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #005489 0%, #0077c2 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 4px 16px rgba(0,84,137,0.2)'
                  }}>
                    <RobotOutlined style={{ fontSize: 40, color: '#fff' }} />
                  </div>
                  <Space direction="vertical" align="center" size={8}>
                    <Title level={3} style={{ margin: 0, color: '#333', fontSize: 28 }}>
                      Meeting Notes AI Assistant
                    </Title>
                    <Text type="secondary" style={{ fontSize: 18, textAlign: 'center' }}>
                      Ask me anything about your meeting notes, factsheets, and transcripts
                    </Text>
                  </Space>
                  <Space direction="vertical" align="start" size={6} style={{ marginTop: 8 }}>
                    <Text type="secondary" style={{ fontSize: 16 }}>
                      Try asking:
                    </Text>
                    <Text type="secondary" style={{ fontSize: 16, paddingLeft: 8 }}>
                      • "What were the key takeaways from recent meetings?"
                    </Text>
                    <Text type="secondary" style={{ fontSize: 16, paddingLeft: 8 }}>
                      • "Summarize the latest factsheet comments"
                    </Text>
                    <Text type="secondary" style={{ fontSize: 16, paddingLeft: 8 }}>
                      • "What topics were discussed about [fund name]?"
                    </Text>
                  </Space>
                </Space>
              </div>
            ) : (
              <Space direction="vertical" style={{ width: '100%', maxWidth: 900, margin: '0 auto' }} size={16}>
                {conversation.map((msg, index) => (
                  <div key={index} style={{ width: '100%' }}>
                    {/* User Question */}
                    <div style={{
                      display: 'flex',
                      justifyContent: 'flex-end',
                      marginBottom: 16
                    }}>
                      <div style={{
                        maxWidth: '70%',
                        background: 'linear-gradient(135deg, #0066a1 0%, #0088cc 100%)',
                        color: '#fff',
                        padding: '16px 20px',
                        borderRadius: '20px 20px 6px 20px',
                        boxShadow: '0 4px 12px rgba(0,102,161,0.15), 0 2px 4px rgba(0,0,0,0.05)',
                        transition: 'transform 0.2s, box-shadow 0.2s'
                      }}>
                        <Text style={{
                          color: '#fff',
                          fontSize: 16,
                          lineHeight: 1.7,
                          fontWeight: 400
                        }}>
                          {msg.question}
                        </Text>
                      </div>
                    </div>

                    {/* AI Response */}
                    <div style={{
                      display: 'flex',
                      justifyContent: 'flex-start',
                      alignItems: 'flex-start'
                    }}>
                      <div style={{
                        width: 38,
                        height: 38,
                        borderRadius: '50%',
                        background: 'linear-gradient(135deg, #f0f4f8 0%, #e5ecf3 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginRight: 12,
                        flexShrink: 0,
                        border: '1px solid #e1e4e8',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.04)'
                      }}>
                        <RobotOutlined style={{ fontSize: 18, color: '#0066a1' }} />
                      </div>
                      <div style={{
                        maxWidth: 'calc(80% - 50px)',
                        background: '#ffffff',
                        padding: '18px 22px',
                        borderRadius: '6px 20px 20px 20px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.03)',
                        border: '1px solid #e8eaed',
                        transition: 'box-shadow 0.2s'
                      }}>
                        {/* Sources */}
                        <div style={{ marginBottom: 14 }}>
                          <Space size={8} wrap>
                            {msg.sources.map((source, idx) => (
                              <span key={idx} style={{
                                fontSize: 12,
                                color: '#6b7280',
                                background: '#f3f4f6',
                                padding: '5px 12px',
                                borderRadius: 12,
                                border: '1px solid #e5e7eb',
                                fontWeight: 500,
                                letterSpacing: '0.2px'
                              }}>
                                {source}
                              </span>
                            ))}
                            <Text type="secondary" style={{
                              fontSize: 12,
                              color: '#9ca3af',
                              fontWeight: 500
                            }}>
                              • {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </Text>
                          </Space>
                        </div>
                        {/* Answer */}
                        <div
                          className="markdown-content"
                          style={{
                            lineHeight: '1.7',
                            color: '#333',
                            fontSize: 16
                          }}
                        >
                          <ReactMarkdown
                            components={{
                              p: ({node, ...props}: any) => <p style={{ margin: '0 0 12px 0', fontSize: 16 }} {...props} />,
                              ul: ({node, ...props}: any) => <ul style={{ margin: '8px 0', paddingLeft: '20px', fontSize: 16 }} {...props} />,
                              ol: ({node, ...props}: any) => <ol style={{ margin: '8px 0', paddingLeft: '20px', fontSize: 16 }} {...props} />,
                              li: ({node, ...props}: any) => <li style={{ margin: '4px 0', fontSize: 16 }} {...props} />,
                              strong: ({node, ...props}: any) => <strong style={{ color: '#005489', fontWeight: 600, fontSize: 16 }} {...props} />,
                              code: ({node, inline, ...props}: any) =>
                                inline
                                  ? <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: 4, fontSize: 15, color: '#d63384' }} {...props} />
                                  : <code style={{ display: 'block', background: '#f5f5f5', padding: '12px', borderRadius: 8, fontSize: 15, overflowX: 'auto' }} {...props} />
                            }}
                          >
                            {msg.answer}
                          </ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </Space>
            )}
          </div>

          {/* Input Area - Fixed at bottom */}
          <div style={{
            background: 'linear-gradient(to top, #ffffff 0%, #fafbfc 100%)',
            borderTop: '1px solid #e1e4e8',
            padding: '24px',
            boxShadow: '0 -4px 16px rgba(0,0,0,0.04)'
          }}>
            <div style={{ maxWidth: 900, margin: '0 auto' }}>
              <div style={{
                background: '#ffffff',
                border: '2px solid #e1e4e8',
                borderRadius: 16,
                padding: '6px',
                display: 'flex',
                alignItems: 'flex-end',
                transition: 'all 0.3s',
                boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
              }}>
                <TextArea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={dataSources.length === 0 ? "Select data sources to start..." : "Ask me anything..."}
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  onPressEnter={(e) => {
                    if (e.shiftKey) return;
                    e.preventDefault();
                    handleAsk();
                  }}
                  style={{
                    resize: 'none',
                    fontSize: 16,
                    border: 'none',
                    background: 'transparent',
                    padding: '14px 18px',
                    fontWeight: 400,
                    lineHeight: 1.6
                  }}
                  disabled={loading || dataSources.length === 0}
                />
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleAsk}
                  loading={loading}
                  style={{
                    background: loading ? '#d1d5db' : (question.trim() && dataSources.length > 0 ? 'linear-gradient(135deg, #0066a1 0%, #0088cc 100%)' : '#d1d5db'),
                    borderColor: 'transparent',
                    height: 44,
                    width: 44,
                    borderRadius: 12,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    marginLeft: 10,
                    boxShadow: (question.trim() && dataSources.length > 0 && !loading) ? '0 4px 12px rgba(0,102,161,0.25)' : 'none',
                    transition: 'all 0.2s',
                    transform: (question.trim() && dataSources.length > 0 && !loading) ? 'scale(1)' : 'scale(0.95)'
                  }}
                  disabled={!question.trim() || dataSources.length === 0}
                />
              </div>
              {dataSources.length === 0 && (
                <Text type="secondary" style={{
                  fontSize: 14,
                  marginTop: 12,
                  display: 'block',
                  textAlign: 'center',
                  color: '#9ca3af',
                  fontWeight: 500
                }}>
                  Select at least one data source from the sidebar to begin
                </Text>
              )}
            </div>
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};

export default ChatBot;
