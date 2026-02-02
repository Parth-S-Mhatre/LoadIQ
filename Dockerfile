# Use official slim Python image (Linux)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY Backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY Backend/ .

# Copy Data artifacts for XGBoost model
COPY DATA_preprocessing /DATA_preprocessing

# Expose port (FastAPI defaults to 8000)
EXPOSE 8000

# Run the application (Default command, can be overridden)
CMD ["uvicorn", "model2:app", "--host", "0.0.0.0", "--port", "8000"]
