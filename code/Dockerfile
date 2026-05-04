FROM python:3.10-slim

# System dependencies for LightGBM and matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY run_pipeline.py /app/run_pipeline.py

# So 'src' is importable as a package
ENV PYTHONPATH="/app"
