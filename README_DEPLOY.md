# Deployment Guide

## Local Development
1. Install Ollama: https://ollama.ai
2. Pull models:
   ollama pull codellama
   ollama pull qwen2.5-coder
3. Install dependencies:
   pip install -r requirements.txt
4. Run:
   python run.py
5. Open:
   API  → http://localhost:8000/docs
   UI   → http://localhost:8501

## Hugging Face Spaces Deployment

### What runs on HF Spaces (free):
- Streamlit frontend only
- Calls your local Ollama via ngrok tunnel

### Steps:
1. Install ngrok: https://ngrok.com
2. Expose your local API:
   ngrok http 8000
3. Copy the ngrok URL (e.g. https://abc123.ngrok.io)
4. In frontend/app.py change:
   API_BASE = "https://abc123.ngrok.io"
5. Push frontend/app.py + requirements.txt to HF Space
6. Space URL becomes your public demo link

### Render Deployment (full backend):
- Connect GitHub repo to Render
- Set build command: pip install -r rag-core/requirements.txt
- Set start command: cd rag-core && uvicorn api.main:app --host 0.0.0.0 --port 8000
- Note: Ollama must run locally, exposed via ngrok
