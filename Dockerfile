# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Don't copy code - we'll mount it
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install dependencies including dev extras
RUN pip install -e ".[dev]"

# Expose port
EXPOSE 8000

# Command to run the application with reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Include health check endpoint
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1