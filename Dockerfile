# ============================================================
# Dockerfile - Amazon Sales Predictive API
# ============================================================
# Build:  docker build -t amazon-sales-api .
# Run:    docker run -p 8000:8000 amazon-sales-api
# Docs:   http://localhost:8000/docs
# ============================================================

# --- Stage 1: Base image ---
# Use a slim Python image to keep container size small
FROM python:3.12-slim

# --- Stage 2: Environment setup ---
# Prevent Python from buffering stdout/stderr (shows logs immediately)
ENV PYTHONUNBUFFERED=1
# Prevent pip from caching download files inside the container
ENV PIP_NO_CACHE_DIR=1

# --- Stage 3: Working directory ---
WORKDIR /app

# --- Stage 4: Install dependencies ---
RUN pip install --upgrade pip && \
    pip install \
    fastapi==0.110.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.6.4 \
    joblib==1.3.2 \
    pandas==2.2.1 \
    numpy==1.26.4 \
    scikit-learn==1.8.0 \
    xgboost==2.0.3 \
    tensorflow==2.16.1

# --- Stage 5: Copy application code ---
# Copy the FastAPI app
COPY Sales_prediction_app.py .

# Copy the serialized model artifacts produced by running the notebook
# (The /models folder must exist locally before building the image)
COPY models/ ./models/

# --- Stage 6: Expose port and boot ---
# FastAPI/Uvicorn listens on port 8000 by default
EXPOSE 8000

# Launch the Uvicorn ASGI server
CMD ["uvicorn", "Sales_prediction_app:app", "--host", "0.0.0.0", "--port", "8000"]
