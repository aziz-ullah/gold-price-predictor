# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies first (better caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend folder
COPY backend/ /app/backend/

# Copy raw data (if not included in backend/)
COPY backend/data/raw/ /app/data/raw/

# Create directories for processed data and models
RUN mkdir -p /app/data/processed /app/models

# Set PYTHONPATH so Python can find backend package
ENV PYTHONPATH=/app

# Run preprocessing
RUN python -m backend.services.preprocess

# Run training (without skipping)
RUN python -m backend.services.trainer --no-tune

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
