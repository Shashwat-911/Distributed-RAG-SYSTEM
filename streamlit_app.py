"""
Streamlit Cloud Root Entry Point for DistributedRAG.
"""
import os
import sys

# Ensure root directory is in Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load and execute frontend/app.py
app_path = os.path.join(ROOT_DIR, "frontend", "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(code, globals())
