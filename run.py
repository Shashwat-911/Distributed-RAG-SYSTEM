"""
Single entry point to launch the full DistributedRAG system.
Usage: python run.py
"""
import subprocess
import sys
import os
import time
import threading

API_DIR = os.path.join(os.path.dirname(__file__), "api")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

API_PORT = os.environ.get("RAG_API_PORT", "8000")
UI_PORT = os.environ.get("RAG_UI_PORT", "8501")

def launch_api():
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", API_PORT,
        "--reload"
    ], cwd=os.path.dirname(__file__))

def launch_frontend():
    time.sleep(3)
    subprocess.run([
        sys.executable, "-m", "streamlit", 
        "run", 
        os.path.join(FRONTEND_DIR, "app.py"),
        "--server.port", UI_PORT,
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    print("Starting DistributedRAG...")
    print(f"API  -> http://localhost:{API_PORT}")
    print(f"UI   -> http://localhost:{UI_PORT}")
    print(f"Docs -> http://localhost:{API_PORT}/docs")
    
    api_thread = threading.Thread(target=launch_api, daemon=True)
    api_thread.start()
    
    launch_frontend()
