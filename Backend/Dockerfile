# Use official slim Python image (Linux)
FROM python:3.11-slim

# Set working directory
WORKDIR /app
ENV MPLCONFIGDIR=/tmp/matplotlib

# Install runtime libs required by LightGBM/XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose port
EXPOSE 8000

# Run the app
# Use JSON form and shell expansion for dynamic PORT.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
