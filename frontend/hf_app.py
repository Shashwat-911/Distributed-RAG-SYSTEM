"""
Hugging Face Spaces entry point for DistributedRAG.
Forwards environment variables and executes the main Streamlit application.
"""
import os
import sys

# Ensure directory is on sys.path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

# Propagate API_BASE to RAG_API_BASE if set in Space secrets
if "API_BASE" in os.environ and "RAG_API_BASE" not in os.environ:
    os.environ["RAG_API_BASE"] = os.environ["API_BASE"]

app_path = os.path.join(_dir, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    exec(f.read(), globals())
