FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000
EXPOSE 8501

CMD ["python", "run.py"]
