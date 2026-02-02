# Use official slim Python image (Linux)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose port
EXPOSE 8000

# Run the app
# Use shell form to allow variable expansion for PORT
CMD sh -c "uvicorn model2:app --host 0.0.0.0 --port ${PORT:-8000}"