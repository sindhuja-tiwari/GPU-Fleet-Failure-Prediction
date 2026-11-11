FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
# Pretrained bundle (fleet 1 of the benchmark), so the demo starts without training.
COPY models/demo-bundle ./models/demo-bundle

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["gpufleet", "serve", "--bundle", "/models/bundle", "--host", "0.0.0.0", "--port", "8000"]
