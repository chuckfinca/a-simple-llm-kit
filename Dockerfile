# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app
COPY . .

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -e ".[dev]"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Include health check endpoint
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1