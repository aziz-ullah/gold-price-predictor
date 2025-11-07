# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models

# Expose port
EXPOSE 8000

# Run preprocessing and training during build
RUN python -m backend.services.preprocess || true
RUN python -m backend.services.trainer --no-tune || true

# Start the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]