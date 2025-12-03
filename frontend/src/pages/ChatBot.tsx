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
  App,
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
  RobotOutlined,
  LoadingOutlined,
  PlusOutlined,
  HistoryOutlined,
  SearchOutlined,
  GlobalOutlined,
  StopOutlined,
  CopyOutlined,
  CheckOutlined
} from '@ant-design/icons';
import { API_ENDPOINTS } from '../config/api';
import dayjs, { Dayjs } from 'dayjs';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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

interface Conversation {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

const ChatBot = () => {
  // Get message API from App context
  const { message } = App.useApp();

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Session ID management - stored in localStorage for persistence
  const [sessionId] = useState<string>(() => {
    const stored = localStorage.getItem('chatbot-session-id');
    if (stored) return stored;
    const newId = crypto.randomUUID();
    localStorage.setItem('chatbot-session-id', newId);
    return newId;
  });

  // State management
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(false);
  const [searchStatus, setSearchStatus] = useState<string>(''); // Search process status

  // Conversation history state
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  // Web search toggle
  const [useWebSearch, setUseWebSearch] = useState(false);

  // Model selection state
  const [model, setModel] = useState<'gemini-2.5-flash' | 'gemini-2.5-pro' | 'gemini-3-low-thinking' | 'gemini-3-high-thinking'>('gemini-3-high-thinking');
  const [thinkingSummary, setThinkingSummary] = useState<string>(''); // Stores thinking progress
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null); // Track which message was copied

  // Filter states
  // Note: Only Meeting Notes has RAG implementation. Factsheet and Transcripts are disabled for now.
  const [dataSources, setDataSources] = useState<string[]>(['Meeting Notes', 'Factsheet']);
  const [useDateFilter, setUseDateFilter] = useState(true);
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs]>([dayjs().subtract(6, 'month').startOf('month'), dayjs().endOf('month')]);
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

  // Load conversations list on mount
  useEffect(() => {
    loadConversations();
  }, [sessionId]);

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

  const loadConversations = async () => {
    try {
      const response = await fetch(`${API_ENDPOINTS.conversationsList}?session_id=${sessionId}`);
      const data = await response.json();
      if (data.success) {
        setConversations(data.conversations);
        // Don't auto-load conversations - let user start fresh or manually select
      }
    } catch (error) {
      console.error('Error loading conversations:', error);
    }
  };

  const loadConversation = async (conversationId: string) => {
    try {
      const response = await fetch(`${API_ENDPOINTS.conversationsGet(conversationId)}?session_id=${sessionId}`);
      const data = await response.json();
      if (data.success && data.conversation) {
        setCurrentConversationId(conversationId);
        const loadedMessages: ChatMessage[] = data.conversation.messages.map((msg: any) => ({
          question: msg.question,
          answer: msg.answer,
          sources: msg.sources,
          timestamp: new Date(msg.timestamp)
        }));
        setConversation(loadedMessages);
      }
    } catch (error) {
      console.error('Error loading conversation:', error);
      message.error('Failed to load conversation');
    }
  };

  const createNewConversation = async (): Promise<string | null> => {
    console.log('Creating new conversation...');
    try {
      const response = await fetch(API_ENDPOINTS.conversationsCreate, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, title: "New Conversation" })
      });
      console.log('Create conversation response:', response);
      const data = await response.json();
      console.log('Create conversation data:', data);
      if (data.success) {
        setCurrentConversationId(data.conversation_id);
        setConversation([]);
        await loadConversations();
        return data.conversation_id;  // Return the ID
      } else {
        message.error('Failed to create conversation');
        return null;
      }
    } catch (error) {
      console.error('Error creating conversation:', error);
      message.error('Failed to create new conversation');
      return null;
    }
  };

  const deleteConversation = async (conversationId: string) => {
    try {
      await fetch(`${API_ENDPOINTS.conversationsDelete(conversationId)}?session_id=${sessionId}`, {
        method: 'DELETE'
      });
      await loadConversations();
      if (conversationId === currentConversationId) {
        setCurrentConversationId(null);
        setConversation([]);
      }
      message.success('Conversation deleted');
    } catch (error) {
      console.error('Error deleting conversation:', error);
      message.error('Failed to delete conversation');
    }
  };

  const saveChatMessage = async (chatMessage: ChatMessage) => {
    if (!currentConversationId) return;

    try {
      await fetch(API_ENDPOINTS.conversationsSaveMessage(currentConversationId), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: {
            question: chatMessage.question,
            answer: chatMessage.answer,
            sources: chatMessage.sources,
            timestamp: chatMessage.timestamp.toISOString()
          }
        })
      });
      await loadConversations(); // Refresh list to update titles and timestamps
    } catch (error) {
      console.error('Error saving chat message:', error);
    }
  };

  // Copy response to clipboard (preserves full HTML with all styling)
  const handleCopyResponse = async (text: string, index: number) => {
    try {
      // Find the rendered HTML content from the DOM
      const messageElements = document.querySelectorAll('.markdown-content');
      const targetElement = messageElements[index];

      if (targetElement) {
        // Get the full HTML with all inline styles preserved
        const htmlContent = targetElement.innerHTML;

        // Create a ClipboardItem with both HTML and plain text
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const plainBlob = new Blob([targetElement.textContent || ''], { type: 'text/plain' });

        await navigator.clipboard.write([
          new ClipboardItem({
            'text/html': blob,
            'text/plain': plainBlob
          })
        ]);
      } else {
        // Fallback to plain text
        await navigator.clipboard.writeText(text);
      }

      setCopiedIndex(index);
      message.success('Copied to clipboard');
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (error) {
      // Fallback for browsers that don't support ClipboardItem
      try {
        await navigator.clipboard.writeText(text);
        setCopiedIndex(index);
        message.success('Copied to clipboard');
        setTimeout(() => setCopiedIndex(null), 2000);
      } catch (e) {
        message.error('Failed to copy');
      }
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

    // Create new conversation if none exists
    let activeConversationId = currentConversationId;
    if (!currentConversationId) {
      const newId = await createNewConversation();
      if (!newId) {
        message.error('Failed to create conversation');
        return;
      }
      activeConversationId = newId;
    }

    setLoading(true);
    setSearchStatus(''); // Start with no status, wait for backend signal
    setThinkingSummary(''); // Clear thinking summary from previous request
    const currentQuestion = question;
    setQuestion(''); // Clear input immediately

    // Build data sources array including web search if enabled
    const effectiveDataSources = useWebSearch ? [...dataSources, 'Web Search'] : dataSources;

    // Create a temporary message with empty answer for streaming
    const tempMessage: ChatMessage = {
      question: currentQuestion,
      answer: '',
      sources: effectiveDataSources,
      timestamp: new Date()
    };

    setConversation(prev => [...prev, tempMessage]);

    // Save the question immediately (with empty answer) so it appears in history
    // This ensures the conversation shows up even if the response takes a long time
    if (activeConversationId) {
      try {
        await fetch(API_ENDPOINTS.conversationsSaveMessage(activeConversationId), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message: {
              question: currentQuestion,
              answer: '',  // Empty answer - will be updated when response completes
              sources: effectiveDataSources,
              timestamp: tempMessage.timestamp.toISOString()
            }
          })
        });
        await loadConversations(); // Refresh list to show the new conversation
      } catch (error) {
        console.error('Error saving initial message:', error);
      }
    }

    // Create AbortController for this request
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const response = await fetch(API_ENDPOINTS.chatAskStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: currentQuestion,
          data_sources: effectiveDataSources,
          start_date: useDateFilter ? dateRange[0].format('YYYY-MM-DD') : null,
          end_date: useDateFilter ? dateRange[1].format('YYYY-MM-DD') : null,
          selected_funds: selectedFunds.length > 0 ? selectedFunds : null,
          conversation_history: conversation.map(msg => ({
            question: msg.question,
            answer: msg.answer
          })),
          model: model
        }),
        signal: abortController.signal
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

              // Handle search status event
              if (jsonData.hasOwnProperty('searching')) {
                if (jsonData.searching) {
                  setSearchStatus('Finding relevant information...');
                }
                // Don't clear on searching: false - let 'generating' event handle the transition
              }

              // Handle generating status event
              if (jsonData.hasOwnProperty('generating')) {
                if (jsonData.generating) {
                  setSearchStatus('Generating response...');
                }
              }

              // Handle thinking event
              if (jsonData.hasOwnProperty('thinking')) {
                setThinkingSummary(jsonData.thinking);
              }

              // Handle heartbeat event (keep-alive for Azure)
              // Just acknowledge - no UI action needed
              if (jsonData.hasOwnProperty('heartbeat')) {
                console.log('[Heartbeat] Connection keep-alive received');
                continue; // Skip to next iteration
              }

              if (jsonData.content) {
                // Clear search status and thinking summary when first content arrives
                if (searchStatus) {
                  setSearchStatus('');
                }
                if (thinkingSummary) {
                  setThinkingSummary('');
                }
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
                // Update the last message's answer in database (message was already saved with empty answer)
                if (activeConversationId) {
                  try {
                    await fetch(API_ENDPOINTS.conversationsUpdateLastMessage(activeConversationId), {
                      method: 'PUT',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        session_id: sessionId,
                        answer: accumulatedAnswer
                      })
                    });
                  } catch (error) {
                    console.error('Error updating message answer:', error);
                  }
                }
              }

              if (jsonData.error) {
                throw new Error(jsonData.error);
              }
            } catch (e) {
              // Ignore JSON parse errors for partial/malformed chunks (SSE can split JSON across chunks)
              if (e instanceof SyntaxError) {
                console.warn('[SSE] Skipping malformed JSON chunk:', line.slice(6, 50) + '...');
                continue; // Skip this line and continue processing
              }
              // Re-throw non-parse errors
              if (e instanceof Error) {
                throw e;
              }
            }
          }
        }
      }
    } catch (error) {
      // Don't show error message if it was aborted by user
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Request was aborted by user');
        return; // handleStop already handled the cleanup
      }
      message.error('Failed to get answer. Please try again.');
      console.error('Error asking question:', error);
      // Remove the temporary message if there was an error
      setConversation(prev => prev.slice(0, -1));
    } finally {
      abortControllerRef.current = null; // Clear the abort controller
      setLoading(false);
      setSearchStatus(''); // Clear search status
      setThinkingSummary(''); // Clear thinking summary
    }
  };

  const handleClearConversation = async () => {
    if (!currentConversationId) {
      setConversation([]);
      return;
    }

    try {
      await deleteConversation(currentConversationId);
      // Optionally create a new conversation immediately
      await createNewConversation();
    } catch (error) {
      console.error('Error clearing conversation:', error);
      message.error('Failed to clear conversation');
    }
  };

  // Stop the current request
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setLoading(false);
      setSearchStatus('');
      setThinkingSummary('');
      // Update the last message to show it was stopped
      setConversation(prev => {
        if (prev.length === 0) return prev;
        const newConv = [...prev];
        const lastMsg = newConv[newConv.length - 1];
        if (!lastMsg.answer) {
          // No answer yet, remove the message entirely
          return prev.slice(0, -1);
        } else {
          // Partial answer, mark it as stopped
          newConv[newConv.length - 1] = {
            ...lastMsg,
            answer: lastMsg.answer + '\n\n*[Response stopped by user]*'
          };
          return newConv;
        }
      });
      message.info('Request stopped');
    }
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
            icon={<PlusOutlined />}
            onClick={createNewConversation}
            type="primary"
            style={{
              borderRadius: 8,
              height: 38,
              padding: '0 16px',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
          >
            New Chat
          </Button>
          <Button
            icon={<HistoryOutlined />}
            onClick={() => setShowHistory(!showHistory)}
            type={showHistory ? "primary" : "default"}
            style={{
              borderRadius: 8,
              height: 38,
              padding: '0 16px',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
          >
            History ({conversations.length})
          </Button>
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
                <Checkbox value="Factsheet" style={{ marginLeft: 0, fontSize: 15 }}>
                  <Text style={{ fontSize: 15 }}>Factsheet</Text>
                </Checkbox>
              </Checkbox.Group>
            </div>

            {/* Date Range - Month Picker */}
            <div style={{ marginBottom: 18 }}>
              <Checkbox
                checked={useDateFilter}
                onChange={(e) => setUseDateFilter(e.target.checked)}
                style={{ marginBottom: 8, fontSize: 15 }}
              >
                <Text style={{ fontSize: 15, color: '#666' }}>Filter by Date</Text>
              </Checkbox>
              {useDateFilter && (
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <DatePicker
                    value={dateRange[0]}
                    onChange={(date) => date && setDateRange([date.startOf('month'), dateRange[1]])}
                    picker="month"
                    format="MMM YYYY"
                    size="middle"
                    placeholder="From"
                    style={{ flex: 1 }}
                  />
                  <Text style={{ color: '#999', fontSize: 13 }}>to</Text>
                  <DatePicker
                    value={dateRange[1]}
                    onChange={(date) => date && setDateRange([dateRange[0], date.endOf('month')])}
                    picker="month"
                    format="MMM YYYY"
                    size="middle"
                    placeholder="To"
                    style={{ flex: 1 }}
                  />
                </div>
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
                maxTagCount={0}
                maxTagPlaceholder={() => (
                  <span style={{ color: '#666' }}>
                    {selectedFunds.length} funds selected
                  </span>
                )}
                showSearch
              />
              {/* Selected funds tags */}
              {selectedFunds.length > 0 && (
                <div style={{
                  marginTop: 8,
                  maxHeight: 120,
                  overflowY: 'auto',
                  padding: '4px 0'
                }}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {selectedFunds.map(fund => (
                      <Tag
                        key={fund}
                        closable
                        onClose={() => setSelectedFunds(selectedFunds.filter(f => f !== fund))}
                        style={{
                          margin: 0,
                          fontSize: 12,
                          padding: '2px 6px',
                          lineHeight: '18px'
                        }}
                      >
                        {fund}
                      </Tag>
                    ))}
                  </div>
                  {selectedFunds.length > 1 && (
                    <Button
                      type="link"
                      size="small"
                      onClick={() => setSelectedFunds([])}
                      style={{ padding: '4px 0', height: 'auto', fontSize: 12 }}
                    >
                      Clear all
                    </Button>
                  )}
                </div>
              )}
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
                  {dataSources.includes('Factsheet') && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={{ fontSize: 15, color: '#666' }}>Factsheet</Text>
                      <Text style={{ fontSize: 15, fontWeight: 500 }}>{filterStats.factsheet_comments_count}</Text>
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

        {/* Conversation History Sidebar */}
        {showHistory && (
          <Sider
            width={280}
            style={{
              background: '#ffffff',
              borderRight: '1px solid #e1e4e8',
              overflow: 'auto'
            }}
          >
            <div style={{ padding: '20px' }}>
              <div style={{
                marginBottom: 16,
                paddingBottom: 12,
                borderBottom: '2px solid #e1e4e8'
              }}>
                <Text strong style={{
                  fontSize: 15,
                  color: '#374151',
                  letterSpacing: '0.5px',
                  textTransform: 'uppercase'
                }}>
                  Chat History
                </Text>
              </div>

              {conversations.length === 0 ? (
                <Empty
                  description="No conversations yet"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  style={{ marginTop: 40 }}
                />
              ) : (
                <Space direction="vertical" style={{ width: '100%' }} size={8}>
                  {conversations.map((conv) => (
                    <div
                      key={conv.conversation_id}
                      onClick={() => loadConversation(conv.conversation_id)}
                      style={{
                        padding: '12px',
                        borderRadius: 8,
                        background: conv.conversation_id === currentConversationId ? '#e6f7ff' : '#f8f9fa',
                        border: `1px solid ${conv.conversation_id === currentConversationId ? '#1890ff' : '#e1e4e8'}`,
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        position: 'relative'
                      }}
                      onMouseEnter={(e) => {
                        if (conv.conversation_id !== currentConversationId) {
                          e.currentTarget.style.background = '#f0f0f0';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (conv.conversation_id !== currentConversationId) {
                          e.currentTarget.style.background = '#f8f9fa';
                        }
                      }}
                    >
                      <div style={{ marginBottom: 6 }}>
                        <Text strong style={{
                          fontSize: 14,
                          color: '#333',
                          display: 'block',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {conv.title}
                        </Text>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {conv.message_count} messages
                        </Text>
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteConversation(conv.conversation_id);
                          }}
                          style={{ padding: '2px 8px' }}
                        />
                      </div>
                    </div>
                  ))}
                </Space>
              )}
            </div>
          </Sider>
        )}

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
                      Ask me anything about your meeting notes and factsheets
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
                  <Space direction="vertical" align="start" size={4} style={{ marginTop: 16, padding: '12px 16px', background: '#f5f5f5', borderRadius: 8, maxWidth: 500 }}>
                    <Text type="secondary" style={{ fontSize: 13, fontStyle: 'italic' }}>
                      Tip: Gemini 3 Thorough can be slow. Use filters for faster results.
                    </Text>
                    <Text type="secondary" style={{ fontSize: 13, fontStyle: 'italic' }}>
                      Note: AI can make mistakes. Please verify important information.
                    </Text>
                  </Space>
                </Space>
              </div>
            ) : (
              <Space direction="vertical" style={{ width: '100%', maxWidth: 1400, margin: '0 auto' }} size={16}>
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
                            remarkPlugins={[remarkGfm]}
                            components={{
                              p: ({node, ...props}: any) => <p style={{ margin: '0 0 12px 0', fontSize: 16 }} {...props} />,
                              ul: ({node, ...props}: any) => <ul style={{ margin: '8px 0', paddingLeft: '20px', fontSize: 16 }} {...props} />,
                              ol: ({node, ...props}: any) => <ol style={{ margin: '8px 0', paddingLeft: '20px', fontSize: 16 }} {...props} />,
                              li: ({node, ...props}: any) => <li style={{ margin: '4px 0', fontSize: 16 }} {...props} />,
                              strong: ({node, ...props}: any) => <strong style={{ color: '#005489', fontWeight: 600, fontSize: 16 }} {...props} />,
                              a: ({node, ...props}: any) => <a style={{ color: '#005489', textDecoration: 'underline' }} target="_blank" rel="noopener noreferrer" {...props} />,
                              code: ({node, inline, ...props}: any) =>
                                inline
                                  ? <code style={{ background: '#f5f5f5', padding: '2px 6px', borderRadius: 4, fontSize: 15, color: '#d63384' }} {...props} />
                                  : <code style={{ display: 'block', background: '#f5f5f5', padding: '12px', borderRadius: 8, fontSize: 15, overflowX: 'auto' }} {...props} />,
                              table: ({node, ...props}: any) => <table style={{ borderCollapse: 'collapse', width: '100%', margin: '16px 0', fontSize: 15 }} {...props} />,
                              thead: ({node, ...props}: any) => <thead style={{ background: '#f5f5f5' }} {...props} />,
                              th: ({node, ...props}: any) => <th style={{ border: '1px solid #ddd', padding: '12px', textAlign: 'left', fontWeight: 600 }} {...props} />,
                              td: ({node, ...props}: any) => <td style={{ border: '1px solid #ddd', padding: '12px' }} {...props} />
                            }}
                          >
                            {msg.answer}
                          </ReactMarkdown>
                        </div>
                        {/* Copy Button */}
                        {msg.answer && (
                          <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
                            <Button
                              type="text"
                              size="small"
                              icon={copiedIndex === index ? <CheckOutlined /> : <CopyOutlined />}
                              onClick={() => handleCopyResponse(msg.answer, index)}
                              style={{
                                color: '#8c8c8c',
                                fontSize: 12
                              }}
                            >
                              {copiedIndex === index ? 'Copied' : 'Copy'}
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}

                {/* Search Status Indicator */}
                {searchStatus && (
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
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12
                    }}>
                      <LoadingOutlined style={{ fontSize: 16, color: '#0066a1' }} />
                      <Text style={{
                        color: '#6b7280',
                        fontSize: 15,
                        fontStyle: 'italic'
                      }}>
                        {searchStatus}
                      </Text>
                    </div>
                  </div>
                )}

                {/* Thinking Summary Indicator */}
                {thinkingSummary && (
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
                      background: '#fef9f3',
                      padding: '18px 22px',
                      borderRadius: '6px 20px 20px 20px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.03)',
                      border: '1px solid #fde4c7',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 12
                    }}>
                      <LoadingOutlined style={{ fontSize: 16, color: '#d97706', marginTop: 2 }} />
                      <div style={{ flex: 1 }}>
                        <Text strong style={{
                          color: '#92400e',
                          fontSize: 13,
                          display: 'block',
                          marginBottom: 4
                        }}>
                          Thinking...
                        </Text>
                        <Text style={{
                          color: '#78716c',
                          fontSize: 14,
                          fontStyle: 'italic',
                          lineHeight: 1.5
                        }}>
                          {thinkingSummary}
                        </Text>
                      </div>
                    </div>
                  </div>
                )}

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
              {/* Web Search Toggle & Model Selector */}
              <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Button
                  size="small"
                  icon={<GlobalOutlined />}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setUseWebSearch(!useWebSearch);
                  }}
                  type="default"
                  style={{
                    borderRadius: 8,
                    border: useWebSearch ? '1px solid #005489' : '1px solid #d1d5db',
                    background: useWebSearch ? '#e6f2f8' : 'transparent',
                    color: useWebSearch ? '#005489' : '#6b7280',
                    fontSize: 13,
                    height: 28,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    fontWeight: useWebSearch ? 500 : 400,
                    transition: 'all 0.2s'
                  }}
                >
                  Allow Web Search
                </Button>

                <Select
                  size="small"
                  value={model}
                  onChange={(value) => setModel(value as 'gemini-2.5-flash' | 'gemini-2.5-pro' | 'gemini-3-low-thinking' | 'gemini-3-high-thinking')}
                  style={{
                    width: 200,
                    borderRadius: 8,
                    fontSize: 13,
                    height: 28
                  }}
                  options={[
                    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
                    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
                    { value: 'gemini-3-low-thinking', label: 'Gemini 3 (Fast)' },
                    { value: 'gemini-3-high-thinking', label: 'Gemini 3 (Thorough)' }
                  ]}
                />
              </div>

              <div style={{
                background: '#ffffff',
                border: '2px solid #e1e4e8',
                borderRadius: 16,
                padding: '6px',
                display: 'flex',
                alignItems: 'center',
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
                {/* Stop/Send button */}
                {loading ? (
                  <Button
                    type="primary"
                    onClick={handleStop}
                    style={{
                      background: '#1a1a1a',
                      borderColor: 'transparent',
                      height: 44,
                      width: 44,
                      borderRadius: 22,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                      marginLeft: 10,
                      boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                      transition: 'all 0.2s'
                    }}
                  >
                    <div style={{
                      width: 14,
                      height: 14,
                      backgroundColor: '#ffffff',
                      borderRadius: 2
                    }} />
                  </Button>
                ) : (
                  <Button
                    type="primary"
                    icon={<SendOutlined />}
                    onClick={handleAsk}
                    style={{
                      background: (question.trim() && dataSources.length > 0) ? 'linear-gradient(135deg, #0066a1 0%, #0088cc 100%)' : '#d1d5db',
                      borderColor: 'transparent',
                      height: 44,
                      width: 44,
                      borderRadius: 12,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                      marginLeft: 10,
                      boxShadow: (question.trim() && dataSources.length > 0) ? '0 4px 12px rgba(0,102,161,0.25)' : 'none',
                      transition: 'all 0.2s',
                      transform: (question.trim() && dataSources.length > 0) ? 'scale(1)' : 'scale(0.95)'
                    }}
                    disabled={!question.trim() || dataSources.length === 0}
                  />
                )}
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
