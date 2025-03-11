# Use official Python image
FROM python:3.10-slim

# Set working direct    ory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the application code
COPY . .

# Set working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--access-log"] 