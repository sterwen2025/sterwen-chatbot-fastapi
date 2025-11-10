# ==============================================================================
# CHATBOT - COMBINED FRONTEND + BACKEND DOCKERFILE
# ==============================================================================
# Single container serving both React frontend and FastAPI backend
# nginx routes: /api/* → backend:8080, /* → frontend static files

# ------------------------------------------------------------------------------
# STAGE 1: Build Frontend
# ------------------------------------------------------------------------------
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./
RUN npm install

# Copy and build frontend
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# STAGE 2: Setup Backend
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS backend-setup

WORKDIR /app/backend

# Copy backend requirements and install
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# ------------------------------------------------------------------------------
# STAGE 3: Final Production Image (Python base with nginx installed)
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Install nginx and supervisor
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from backend-setup stage
COPY --from=backend-setup /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-setup /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy backend application
COPY --from=backend-setup /app/backend /app/backend

# Copy frontend build
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/sites-available/default
RUN rm -f /etc/nginx/sites-enabled/default && \
    ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/ && \
    # Disable daemon mode for supervisor
    echo "daemon off;" >> /etc/nginx/nginx.conf

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create nginx directories and set permissions
RUN mkdir -p /var/log/nginx /var/lib/nginx /run && \
    chown -R www-data:www-data /var/log/nginx /var/lib/nginx /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start supervisor (manages nginx + uvicorn)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
