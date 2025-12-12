# -------------------------------
# FASTAPI + UVICORN BACKEND DOCKERFILE
# -------------------------------
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port for Render / Docker deployment
EXPOSE 8000

# Run FastAPI using start script
CMD ["sh", "start.sh"]
