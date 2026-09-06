# Deployment Guide

## Local Development
1. Install Ollama: https://ollama.ai
2. Pull models:
   ```bash
   ollama pull codellama
   ollama pull qwen2.5-coder
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python run.py
   ```
5. Open:
   - API  → `http://localhost:8000/docs`
   - UI   → `http://localhost:8501`

---

## Deployed Frontend (Hugging Face Spaces / Streamlit Cloud)

The Streamlit UI can run in the cloud on Hugging Face Spaces or Streamlit Cloud while connecting to your FastAPI backend (via tunnel or cloud host).

### Connecting to Your Backend (3 Easy Ways — No Code Edits Required!)

1. **In-UI Sidebar (Fastest)**:
   - Run your local API: `python run.py`
   - Start a tunnel in your terminal:
     ```bash
     # Using untun (Cloudflare tunnel without account):
     npx untun tunnel 8000

     # OR using ngrok:
     ngrok http 8000
     ```
   - Copy the public `https://...` URL.
   - In your deployed Streamlit app, paste the URL in the sidebar **Backend API URL** box and click **⚡ Test Ping**.

2. **URL Query Parameter (Best for sharing demos)**:
   - Append `?api=https://your-tunnel-url.com` to your Hugging Face or Streamlit Cloud URL.
   - Example: `https://huggingface.co/spaces/user/rag?api=https://abc-123.ngrok-free.app`

3. **Space / Streamlit Secrets (Best for persistent backends)**:
   - Set secret: `RAG_API_BASE = "https://your-backend-url.com"`

---

## Render Deployment (Backend API)
- Connect GitHub repo to Render as a Web Service.
- Build command: `pip install -r rag-core/requirements.txt`
- Start command: `cd rag-core && uvicorn api.main:app --host 0.0.0.0 --port 8000`
- **Render Free Tier Cold Starts**: Render puts free services to sleep after 15 min of inactivity. The new frontend has built-in retry and cold-start detection (click **⚡ Test Ping** in the sidebar to wake up the server).
