# Meeting Notes Chatbot

Intelligent Q&A system for meeting notes, factsheet comments, and transcripts powered by Google Gemini AI.

## Features

- 💬 **Conversational AI**: Ask questions about your meeting notes and fund comments
- 🔍 **Advanced Filtering**: Filter by data sources, date ranges, portfolios, and funds
- 📊 **Live Statistics**: Real-time count of available data based on filters
- 🎯 **Context-Aware**: Maintains conversation history for better responses
- 🎨 **Unified UI**: Consistent design with other Sterwen services

## Architecture

- **Frontend**: React + TypeScript + Ant Design + Vite
- **Backend**: FastAPI + Python 3.11
- **AI**: Google Gemini 2.0 Flash
- **Database**: MongoDB (Azure Cosmos DB)
- **Deployment**: Docker + Azure App Service (Combined container with nginx)

## Local Development

### Prerequisites

- Node.js 18+
- Python 3.11+
- MongoDB connection string

### Setup

1. **Backend**:
```bash
cd backend
pip install -r requirements.txt
export MONGO_URI="your-mongodb-uri"
export GEMINI_API_KEY="your-gemini-api-key"
python -m uvicorn main:app --reload --port 8000
```

2. **Frontend**:
```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`

## Production Deployment

### Build Docker Image

```bash
docker build -t chatbot:latest .
```

### Run Container

```bash
docker run -p 80:80 \
  -e MONGO_URI="your-mongodb-uri" \
  -e GEMINI_API_KEY="your-gemini-api-key" \
  chatbot:latest
```

## Environment Variables

- `MONGO_URI`: MongoDB connection string
- `GEMINI_API_KEY`: Google Gemini API key
- `VITE_API_URL`: API base URL (for development only)

## API Endpoints

- `GET /health`: Health check
- `GET /api/chat/funds`: Get all available funds
- `GET /api/chat/portfolios`: Get all portfolios
- `POST /api/chat/portfolios/funds`: Get funds by portfolios
- `POST /api/chat/stats`: Get live data statistics
- `POST /api/chat/ask`: Ask a question

## Data Sources

1. **Meeting Notes**: Internal meeting notes from fund managers
2. **Factsheet Comments**: Monthly factsheet commentary
3. **Transcripts**: Meeting transcripts from Fireflies.ai

## License

© 2025 Sterwen. All rights reserved.

# Test trigger
