# ==============================================================================
# PRODUCTION DOCKERFILE FOR CHATBOT SERVICE
# ==============================================================================
# FastAPI service for RAG-powered meeting notes chatbot

FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8080 (Azure App Service uses this)
EXPOSE 8080

# Start uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
