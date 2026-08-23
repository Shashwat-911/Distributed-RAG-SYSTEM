"""
Streamlit Web Application for DistributedRAG (rag-core).

Provides a comprehensive 4-tab user interface for:
1. Interactive Chat with source inspection, latency metrics, and cache indicators.
2. Document Management for raw ingestion, delta updates, and deletions.
3. Multi-mode Retrieval Inspector comparing keyword, vector, and hybrid search.
4. Real-time System Diagnostics and LRU cache controls.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Global Configuration & Initialization
# ─────────────────────────────────────────────────────────────────────────────

import os

def _resolve_default_api() -> str:
    # 1. Check Streamlit Cloud secrets
    try:
        if hasattr(st, "secrets"):
            if "RAG_API_BASE" in st.secrets:
                return str(st.secrets["RAG_API_BASE"]).strip()
            if "API_BASE" in st.secrets:
                return str(st.secrets["API_BASE"]).strip()
    except Exception:
        pass
    # 2. Check environment variables
    env_val = os.environ.get("RAG_API_BASE") or os.environ.get("API_BASE")
    if env_val:
        return env_val.strip()
    # 3. Default fallback to active tunnel
    return "https://mail-shadows-slots-las.trycloudflare.com"

DEFAULT_API_BASE = _resolve_default_api()

st.set_page_config(
    page_title="DistributedRAG",
    page_icon="🔍",
    layout="wide",
)

# Custom styling for badges and metric highlights
st.markdown(
    """
    <style>
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-green { background-color: rgba(46, 164, 79, 0.2); color: #2ea44f; border: 1px solid #2ea44f; }
    .badge-orange { background-color: rgba(219, 109, 40, 0.2); color: #db6d28; border: 1px solid #db6d28; }
    .badge-red { background-color: rgba(207, 34, 46, 0.2); color: #cf222e; border: 1px solid #cf222e; }
    .badge-blue { background-color: rgba(9, 105, 218, 0.2); color: #0969da; border: 1px solid #0969da; }
    .status-online { color: #2ea44f; font-size: 1.6rem; font-weight: 700; }
    .status-offline { color: #cf222e; font-size: 1.6rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Safely initialize conversation history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# If old localhost is cached in session state, reset to default tunnel
if st.session_state.get("api_base") in ("http://localhost:8000", "http://localhost:8000/"):
    st.session_state["api_base"] = DEFAULT_API_BASE

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Query Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ RAG Configuration")
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5, step=1)
    mode = st.selectbox("Retrieval Mode", options=["hybrid", "keyword", "vector"], index=0)
    use_cache = st.checkbox("Use Cache", value=True)
    st.divider()
    api_url_input = st.text_input(
        "Backend API URL",
        value=st.session_state.get("api_base", DEFAULT_API_BASE),
        help="Change this to your deployed FastAPI backend URL or tunnel URL",
    )
    API_BASE = api_url_input.rstrip("/") if api_url_input.strip() else DEFAULT_API_BASE
    st.session_state["api_base"] = API_BASE
    st.caption(f"Target API: `{API_BASE}`")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_query_api(query_text: str, k: int, search_mode: str, cache_toggle: bool) -> Optional[Dict[str, Any]]:
    payload = {
        "query": query_text,
        "top_k": k,
        "mode": search_mode,
        "use_cache": cache_toggle,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{API_BASE}/query", json=payload)
            if resp.status_code == 200:
                return resp.json()
            else:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
                st.error(f"API Error ({resp.status_code}): {detail}")
                return None
    except Exception as exc:
        st.error(f"Connection failed: {exc}")
        return None


def _fetch_health() -> Optional[Dict[str, Any]]:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_BASE}/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def _fetch_cache_stats() -> Optional[Dict[str, Any]]:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_BASE}/cache/stats")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Navigation Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_chat, tab_docs, tab_inspector, tab_system = st.tabs([
    "💬 Chat",
    "📄 Documents",
    "🔬 Retrieval Inspector",
    "⚡ System",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Chat
# ═════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.subheader("Interactive RAG Assistant")

    # Render conversation history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**[{idx}] Document `{src.get('doc_id')}`** — Score: `{src.get('score', 0.0):.4f}` ({src.get('source', '')})")
                            st.info(src.get("chunk_text", ""))

                c1, c2, c3 = st.columns(3)
                ret_ms = msg.get("retrieval_time_ms", 0.0)
                gen_ms = msg.get("generation_time_ms", 0.0)
                cached = msg.get("cached", False)
                model_used = msg.get("model_used", "unknown")

                c1.metric("Retrieval Latency", f"{ret_ms:.1f} ms")
                c2.metric("Generation Latency", f"{gen_ms:.1f} ms")
                cache_label = f"⚡ SEMANTIC CACHE ({model_used})" if (cached and "semantic" in model_used) else ("⚡ CACHED" if cached else f"🤖 {model_used}")
                c3.metric("Cache / Model", cache_label)

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the indexed corpus...")
    if user_query:
        # Display and record user message
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Call RAG API
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context & generating answer..."):
                resp = _call_query_api(user_query, top_k, mode, use_cache)

            if resp:
                answer = resp.get("answer", "")
                sources = resp.get("sources", [])
                cached = resp.get("cached", False)
                model_used = resp.get("model_used", "")
                ret_ms = resp.get("retrieval_time_ms", 0.0)
                gen_ms = resp.get("generation_time_ms", 0.0)

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**[{idx}] Document `{src.get('doc_id')}`** — Score: `{src.get('score', 0.0):.4f}` ({src.get('source', '')})")
                            st.info(src.get("chunk_text", ""))

                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval Latency", f"{ret_ms:.1f} ms")
                c2.metric("Generation Latency", f"{gen_ms:.1f} ms")
                cache_label = f"⚡ SEMANTIC CACHE ({model_used})" if (cached and "semantic" in model_used) else ("⚡ CACHED" if cached else f"🤖 {model_used}")
                c3.metric("Cache / Model", cache_label)

                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "cached": cached,
                    "model_used": model_used,
                    "retrieval_time_ms": ret_ms,
                    "generation_time_ms": gen_ms,
                })


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Documents
# ═════════════════════════════════════════════════════════════════════════════

with tab_docs:
    st.subheader("Document Ingestion & Incremental Update")

    doc_id_input = st.text_input("Document ID", placeholder="e.g. architecture_guide.md")
    doc_content_input = st.text_area("Document Content (Markdown or Raw Text)", height=220, placeholder="# Heading\nParagraph content...")

    btn_col1, btn_col2 = st.columns(2)
    ingest_clicked = btn_col1.button("📥 Ingest Document", use_container_width=True)
    update_clicked = btn_col2.button("🔄 Update Document", use_container_width=True)

    if ingest_clicked:
        if not doc_id_input.strip() or not doc_content_input.strip():
            st.error("Both Document ID and Content are required for ingestion.")
        else:
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{API_BASE}/documents/ingest",
                        json={"doc_id": doc_id_input.strip(), "text": doc_content_input},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ Added {data.get('chunks_added', 0)} chunks")
                    else:
                        st.error(f"Ingestion failed ({resp.status_code}): {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    if update_clicked:
        if not doc_id_input.strip() or not doc_content_input.strip():
            st.error("Both Document ID and Content are required for update.")
        else:
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{API_BASE}/documents/update",
                        json={"doc_id": doc_id_input.strip(), "new_text": doc_content_input},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ Document `{data.get('doc_id')}` updated successfully via Hirschberg LCS Diff")
                        m1, m2, m3 = st.columns(3)
                        m1.markdown(
                            f"<div class='metric-badge badge-green'>🟩 Chunks Added: {data.get('chunks_added', 0)}</div>",
                            unsafe_allow_html=True,
                        )
                        m2.markdown(
                            f"<div class='metric-badge badge-orange'>🟧 Chunks Modified: {data.get('chunks_modified', 0)}</div>",
                            unsafe_allow_html=True,
                        )
                        m3.markdown(
                            f"<div class='metric-badge badge-red'>🟥 Chunks Deleted: {data.get('chunks_deleted', 0)}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(f"Update failed ({resp.status_code}): {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    st.divider()

    st.subheader("🗑️ Delete Document")
    del_col1, del_col2 = st.columns([3, 1])
    delete_doc_id = del_col1.text_input("Document ID to delete", placeholder="e.g. architecture_guide.md", key="delete_doc_id_key")
    delete_clicked = del_col2.button("🗑️ Delete Document", use_container_width=True)

    if delete_clicked:
        if not delete_doc_id.strip():
            st.error("Please provide a Document ID to delete.")
        else:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.delete(f"{API_BASE}/documents/{delete_doc_id.strip()}")
                    if resp.status_code == 200:
                        st.success(f"✅ Document `{delete_doc_id.strip()}` deleted successfully.")
                    else:
                        st.error(f"Deletion failed ({resp.status_code}): {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Retrieval Inspector
# ═════════════════════════════════════════════════════════════════════════════

with tab_inspector:
    st.subheader("🔬 Parallel Retrieval Inspector")
    st.caption("Compare Keyword (TF-IDF), Dense Vector (KD-Tree), and Hybrid search results side-by-side.")

    inspect_query = st.text_input("Search query for inspection", placeholder="Enter search keywords or semantics...")
    inspect_btn = st.button("🚀 Run Inspection", type="primary")

    if inspect_btn:
        if not inspect_query.strip():
            st.error("Please enter a search query.")
        else:
            def _fetch_mode_results(search_mode: str) -> List[Dict[str, Any]]:
                payload = {
                    "query": inspect_query.strip(),
                    "top_k": 5,
                    "mode": search_mode,
                    "use_cache": False,
                }
                try:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.post(f"{API_BASE}/query", json=payload)
                        if resp.status_code == 200:
                            return resp.json().get("sources", [])
                except Exception:
                    pass
                return []

            with st.spinner("Executing parallel retrieval across all 3 search modes..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_kw = executor.submit(_fetch_mode_results, "keyword")
                    future_vec = executor.submit(_fetch_mode_results, "vector")
                    future_hyb = executor.submit(_fetch_mode_results, "hybrid")

                    kw_results = future_kw.result()
                    vec_results = future_vec.result()
                    hyb_results = future_hyb.result()

            col_kw, col_vec, col_hyb = st.columns(3)

            # Keyword Column
            with col_kw:
                st.markdown("### 🔤 Keyword (TF-IDF)")
                if not kw_results:
                    st.warning("No keyword matches found.")
                else:
                    for i, src in enumerate(kw_results, 1):
                        score = float(src.get("score", 0.0))
                        st.markdown(f"**#{i} `{src.get('doc_id')}`** — Score: `{score:.4f}`")
                        st.progress(min(max(score, 0.0), 1.0))
                        with st.expander("Preview chunk", expanded=(i == 1)):
                            st.write(src.get("chunk_text", ""))

            # Vector Column
            with col_vec:
                st.markdown("### 📐 Vector (KD-Tree)")
                if not vec_results:
                    st.warning("No vector matches found.")
                else:
                    for i, src in enumerate(vec_results, 1):
                        score = float(src.get("score", 0.0))
                        st.markdown(f"**#{i} `{src.get('doc_id')}`** — Score: `{score:.4f}`")
                        st.progress(min(max(score, 0.0), 1.0))
                        with st.expander("Preview chunk", expanded=(i == 1)):
                            st.write(src.get("chunk_text", ""))

            # Hybrid Column
            with col_hyb:
                st.markdown("### ⚡ Hybrid (Fused)")
                if not hyb_results:
                    st.warning("No hybrid matches found.")
                else:
                    for i, src in enumerate(hyb_results, 1):
                        score = float(src.get("score", 0.0))
                        st.markdown(f"**#{i} `{src.get('doc_id')}`** — Score: `{score:.4f}`")
                        st.progress(min(max(score, 0.0), 1.0))
                        with st.expander("Preview chunk", expanded=(i == 1)):
                            st.write(src.get("chunk_text", ""))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: System
# ═════════════════════════════════════════════════════════════════════════════

with tab_system:
    st.subheader("⚡ System Diagnostics & Cache Controls")

    col_btn, _ = st.columns([2, 4])
    clear_cache_clicked = col_btn.button("🧹 Clear Cache", type="secondary")

    if clear_cache_clicked:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(f"{API_BASE}/cache/clear")
                if resp.status_code == 200:
                    st.success("✅ Cache cleared successfully.")
                else:
                    st.error(f"Failed to clear cache: {resp.text}")
        except Exception as exc:
            st.error(f"Request error: {exc}")

    # Dynamic status container
    status_placeholder = st.empty()

    health_data = _fetch_health()
    cache_data = _fetch_cache_stats()

    with status_placeholder.container():
        st.markdown("#### Health Status")
        if health_data is not None:
            st.markdown("<div class='status-online'>🟢 ONLINE</div>", unsafe_allow_html=True)
            h_c1, h_c2, h_c3 = st.columns(3)
            ollama_on = health_data.get("ollama_connected", False)
            h_c1.markdown(
                f"**Ollama LLM Status:** "
                + (f"<span class='metric-badge badge-green'>CONNECTED</span>" if ollama_on else "<span class='metric-badge badge-red'>DISCONNECTED</span>"),
                unsafe_allow_html=True,
            )
            h_c2.metric("Documents Indexed", health_data.get("docs_indexed", 0))
            h_c3.metric("Active Cache Entries", health_data.get("cache_size", 0))
        else:
            st.markdown("<div class='status-offline'>🔴 OFFLINE (Backend Unreachable)</div>", unsafe_allow_html=True)

        st.divider()

        st.markdown("#### ⚡ AeroCache Distributed Engine Performance")
        if cache_data is not None:
            hit_rate = float(cache_data.get("hit_rate", 0.0))
            hits = int(cache_data.get("hits", 0))
            misses = int(cache_data.get("misses", 0))
            cache_sz = int(cache_data.get("size", 0))

            st.write(f"**Cache Hit Rate:** `{hit_rate * 100:.1f}%` *(Sub-millisecond instant responses)*")
            st.progress(min(max(hit_rate, 0.0), 1.0))

            cs_c1, cs_c2, cs_c3 = st.columns(3)
            cs_c1.metric("Cache Hits (0 ms)", hits)
            cs_c2.metric("Cache Misses", misses)
            cs_c3.metric("AeroCache Entries", cache_sz)

            st.markdown(
                """
                <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-top: 10px;'>
                    <span class='metric-badge badge-blue'>AeroCache 4-Partition Sharding</span>
                    <span class='metric-badge badge-green'>ConsistentHashRing (150 vnodes)</span>
                    <span class='metric-badge badge-orange'>NexusSearch MapReduce TF-IDF + KD-Tree</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Cache statistics unavailable while backend is offline.")
