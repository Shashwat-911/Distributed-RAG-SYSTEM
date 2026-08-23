"""
Hugging Face Spaces entry point.
Identical to app.py but reads API_BASE from environment variable
so it can point to ngrok tunnel or Render backend.
"""
import os
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# Re-export everything from app.py with overridden API_BASE
exec(open(os.path.join(os.path.dirname(__file__), "app.py")).read())
