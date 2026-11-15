"""
app.py - Streamlit web interface for the Course Doubt-Solver.
Run with:  streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag import RAGEngine

load_dotenv()
st.set_page_config(page_title="Course Doubt-Solver", page_icon="📚", layout="wide")
st.title("📚 Course Doubt-Solver")
st.caption("Upload your course PDFs, ask doubts, get answers with document name + page number.")

# ---------------- sidebar: API key, upload, settings ----------------
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input(
        "Groq API key", type="password", value=os.getenv("GROQ_API_KEY", "")
    )
    if not api_key:
        st.info("Get a free key at console.groq.com and paste it here.")
        st.stop()


@st.cache_resource(show_spinner="Loading AI models (first time takes a minute)...")
def get_engine(key: str) -> RAGEngine:
    return RAGEngine(api_key=key)


engine = get_engine(api_key)

with st.sidebar:
    st.header("2. Upload material")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", type="primary") and files:
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                n = engine.add_document(f.name, f.getvalue())
            if n:
                st.success(f"{f.name}: {n} chunks indexed")
            else:
                st.warning(f"{f.name}: no text found (scanned PDF?)")

    st.header("3. Settings")
    docs = engine.list_documents()
    chosen = st.multiselect("Search only in (empty = all)", docs)
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 5)
    min_score = st.slider("Minimum relevance", 0.0, 0.8, 0.2, 0.05,
                          help="If the best match is below this, we say 'not found' instead of guessing.")

    st.header("4. Study tools")
    topic = st.text_input("Topic or chapter", placeholder="e.g. Archimedean property")
    b1, b2 = st.columns(2)
    do_sum = b1.button("Summarize")
    do_quiz = b2.button("Practice Qs")
    if do_sum or do_quiz:
        if not topic.strip():
            st.warning("Type a topic first.")
        elif not docs:
            st.warning("Upload and process a PDF first.")
        else:
            mode = "summary" if do_sum else "quiz"
            with st.spinner("Preparing..."):
                hits = engine.retrieve(topic, max(top_k, 8), chosen or None)
                if not hits or hits[0]["score"] < min_score:
                    reply = "I could not find this topic in your uploaded course material."
                    hits = []
                else:
                    reply = engine.study_tool(mode, topic, hits)
            label = "Summary" if mode == "summary" else "Practice questions"
            st.session_state.setdefault("messages", [])
            st.session_state.messages.append({"role": "user", "content": f"{label}: {topic}"})
            st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})

    st.header("Your documents")
    for d in docs:
        c1, c2 = st.columns([4, 1])
        c1.write(f"📄 {d}")
        if c2.button("🗑", key=f"del-{d}"):
            engine.delete_document(d)
            st.rerun()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(hits):
    with st.expander(f"📎 Sources ({len(hits)} passages used)"):
        for h in hits:
            st.markdown(f"**{h['source']}** — page **{h['page']}** — relevance `{h['score']}`")
            st.caption(h["text"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"])

question = st.chat_input("Ask a doubt from your course material...")
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not docs:
            reply, hits = "Please upload and process at least one PDF first.", []
        else:
            with st.spinner("Searching your notes..."):
                standalone = engine.condense_question(history, question)
                hits = engine.retrieve(standalone, top_k, chosen or None)
                if not hits or hits[0]["score"] < min_score:
                    reply = "I could not find this in your uploaded course material."
                    hits = []
                else:
                    reply = engine.answer(standalone, hits)
        st.markdown(reply)
        if hits:
            show_sources(hits)
    st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})
