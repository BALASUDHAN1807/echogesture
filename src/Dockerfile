FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# inference-only dependencies
COPY requirements.txt .
RUN grep -vE "^(torch|onnx>|tqdm)" requirements.txt > req-infer.txt && pip install --no-cache-dir -r req-infer.txt
COPY . .
CMD ["python", "main.py", "--help"]
