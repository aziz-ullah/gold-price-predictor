# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies (ignore errors for optional packages)
RUN pip install --no-cache-dir -r requirements.txt || \
    (pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn joblib requests && \
     echo "Installed core dependencies, some optional packages may be missing")

# Copy the entire backend directory
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models

# Expose port
EXPOSE 8000

# Run preprocessing and training during build (non-blocking)
RUN python -m backend.services.preprocess || echo "Preprocessing skipped"
RUN python -m backend.services.trainer --no-tune || echo "Training skipped"

# Start the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]