const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const API_ENDPOINTS = {
  chatFunds: `${API_BASE_URL}/chat/funds`,
  chatPortfolios: `${API_BASE_URL}/chat/portfolios`,
  chatPortfoliosFunds: `${API_BASE_URL}/chat/portfolios/funds`,
  chatStats: `${API_BASE_URL}/chat/stats`,
  chatAsk: `${API_BASE_URL}/chat/ask`,
};
