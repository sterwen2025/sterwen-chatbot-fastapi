const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const API_ENDPOINTS = {
  chatFunds: `${API_BASE_URL}/chat/funds`,
  chatPortfolios: `${API_BASE_URL}/chat/portfolios`,
  chatPortfoliosFunds: `${API_BASE_URL}/chat/portfolios/funds`,
  chatStats: `${API_BASE_URL}/chat/stats`,
  chatAsk: `${API_BASE_URL}/chat/ask`,
  chatAskStream: `${API_BASE_URL}/chat/ask/stream`,
  conversationsCreate: `${API_BASE_URL}/chat/conversations/create`,
  conversationsList: `${API_BASE_URL}/chat/conversations`,
  conversationsGet: (id: string) => `${API_BASE_URL}/chat/conversations/${id}`,
  conversationsDelete: (id: string) => `${API_BASE_URL}/chat/conversations/${id}`,
  conversationsSaveMessage: (id: string) => `${API_BASE_URL}/chat/conversations/${id}/messages`,
};
