# Use a small Python image
FROM python:3.11-slim

# System deps (optional but useful for pandas/numpy performance)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Hugging Face Spaces expects the app to run on port 7860
ENV PORT=7860
EXPOSE 7860

# Start the Dash app (your app.py already reads PORT env)
CMD ["python", "app.py"]
