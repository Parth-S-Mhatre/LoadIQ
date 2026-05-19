# Use official slim Python image (Linux)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime libs required by LightGBM/XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY Backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY Backend/ .

# Expose port (FastAPI defaults to 8000)
EXPOSE 8000

# Run the application (Default command, can be overridden)
# Use JSON form and shell expansion for dynamic PORT.
CMD ["sh", "-c", "uvicorn Model2:app --host 0.0.0.0 --port ${PORT:-8000}"]
