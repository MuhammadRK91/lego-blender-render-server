FROM python:3.11-slim

# Install Blender and system dependencies
RUN apt-get update && apt-get install -y \
    blender \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Render uses PORT environment variable
ENV PORT=8000

# Start FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
