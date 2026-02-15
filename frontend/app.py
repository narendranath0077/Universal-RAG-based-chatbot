from __future__ import annotations

import requests
import streamlit as st

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"

st.set_page_config(page_title="Universal Agentic RAG", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at top right, #1a1a1a 0%, #050505 55%, #000000 100%);
    color: #d1d5db;
}
.block-container {padding-top: 1.2rem;}
.glass-card {
    background: rgba(78, 78, 78, 0.18);
    border: 1px solid rgba(199, 199, 199, 0.18);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(12px);
    padding: 14px;
    margin-bottom: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 Universal QA RAG Chatbot")
st.caption("Agentic RAG with FAISS + Hybrid Reranking + Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 📁 Ingestion")
    uploaded_files = st.file_uploader(
        "Upload files or a zip bundle",
        accept_multiple_files=True,
        type=["txt", "md", "pdf", "docx", "csv", "xlsx", "json", "zip"],
    )

    if st.button("Ingest Uploaded Files", use_container_width=True) and uploaded_files:
        files_payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploaded_files]
        try:
            res = requests.post(f"{API_URL}/ingest/upload", files=files_payload, timeout=300)
            if res.ok:
                st.success(res.json().get("message", "Ingestion completed."))
            else:
                st.error(res.text)
        except requests.RequestException as exc:
            st.error(f"Backend unreachable: {exc}")

    st.divider()
    st.markdown("### 🗂️ Ingest by File Paths")
    raw_paths = st.text_area("One absolute path per line")
    if st.button("Ingest Paths", use_container_width=True):
        paths = [p.strip() for p in raw_paths.splitlines() if p.strip()]
        if paths:
            try:
                res = requests.post(f"{API_URL}/ingest/paths", json={"file_paths": paths}, timeout=300)
                if res.ok:
                    st.success(res.json().get("message", "Path ingestion completed."))
                else:
                    st.error(res.text)
            except requests.RequestException as exc:
                st.error(f"Backend unreachable: {exc}")

st.markdown('<div class="glass-card">Ask anything from your indexed data.</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"query": prompt}, timeout=180)
                if response.ok:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    source_text = "\n".join(
                        [f"- {s.get('file_name', 'unknown')} (score={s.get('hybrid_score', 0) or 0:.3f})" for s in sources]
                    )
                    full = f"{answer}\n\n**Sources**\n{source_text if source_text else '- none'}"
                else:
                    full = f"Error: {response.text}"
            except requests.RequestException as exc:
                full = f"Error contacting backend: {exc}"
        st.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
