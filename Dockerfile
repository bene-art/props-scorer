# Props Scorer
# Production-ready ML inference service

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install production dependencies only (no curl needed)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy health check script
COPY scripts/healthcheck.py ./scripts/

# Copy application code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Health check using Python (no external deps required)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/healthcheck.py

# Run with uvicorn
CMD ["uvicorn", "inference_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
