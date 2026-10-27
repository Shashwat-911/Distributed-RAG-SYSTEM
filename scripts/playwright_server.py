"""Run the real Streamlit UI against an explicit in-memory test API.

This fixture validates frontend requests/rendering, not retrieval/LLM correctness.
It is only launched by the Playwright development command.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DOCUMENTS = {}
CACHE = {}
HITS = 0
MISSES = 0

class FixtureAPI(BaseHTTPRequestHandler):
    def reply(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path in ("/", "/health"):
            self.reply({"status":"playwright-fixture","ollama_connected":False,
                        "docs_indexed":len(DOCUMENTS),"cache_size":len(CACHE)})
        elif self.path == "/cache/stats":
            self.reply({"hits":HITS,"misses":MISSES,"hit_rate":HITS/max(1,HITS+MISSES),"size":len(CACHE)})
        else:
            self.reply({"detail":"Fixture route not found"},404)

    def do_POST(self):
        global HITS, MISSES
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path == "/documents/ingest":
            if data["doc_id"] == "reject-fixture":
                self.reply({"detail":"Synthetic ingestion rejection"},400)
                return
            DOCUMENTS[data["doc_id"]] = data["text"]
            self.reply({"doc_id":data["doc_id"],"chunks_added":1,"status":"ingested"})
        elif self.path == "/documents/update":
            if data["doc_id"] not in DOCUMENTS:
                self.reply({"detail":"Document not found"},404)
                return
            DOCUMENTS[data["doc_id"]] = data["new_text"]
            self.reply({"doc_id":data["doc_id"],"chunks_added":0,"chunks_modified":1,"chunks_deleted":0})
        elif self.path == "/query":
            cached = data.get("use_cache", True) and data["query"] in CACHE
            if cached: HITS += 1
            else: MISSES += 1
            result = {"answer":"Fixture answer: browser integration is working.",
                "sources":[{"doc_id":"fixture-guide","chunk_text":"Synthetic source for frontend tests.",
                            "score":1.0,"source":data.get("mode","hybrid")}],
                "cached":cached,"model_used":"playwright-fixture",
                "retrieval_time_ms":1.0,"generation_time_ms":1.0}
            if data.get("use_cache", True): CACHE[data["query"]] = result
            self.reply(result)
        elif self.path == "/cache/clear":
            CACHE.clear()
            HITS = MISSES = 0
            self.reply({"status":"cleared"})
        else:
            self.reply({"detail":"Fixture route not found"},404)

    def do_DELETE(self):
        doc_id = self.path.removeprefix("/documents/")
        if doc_id not in DOCUMENTS:
            self.reply({"detail":"Document not found"},404)
        else:
            del DOCUMENTS[doc_id]
            self.reply({"doc_id":doc_id,"status":"deleted"})

    def log_message(self, *_args):
        pass

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureAPI)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    print(f"PLAYWRIGHT FIXTURE API: {endpoint}; model inference is not exercised.", flush=True)
    env = dict(os.environ, RAG_API_BASE=endpoint, API_BASE=endpoint)
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(root/"frontend"/"app.py"),
        "--server.address", "127.0.0.1", "--server.port", os.getenv("PLAYWRIGHT_PORT","9104"),
        "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=root, env=env)
    try:
        sys.exit(process.wait())
    finally:
        process.terminate()
        server.shutdown()
