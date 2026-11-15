"""
rag.py - the brain of the Course Doubt-Solver.

Pipeline (each step is a function/method below):
  1. load_pdf_pages()  : read a PDF page by page  (so we never lose page numbers)
  2. chunk_text()      : cut each page into small overlapping pieces
  3. embed             : turn each piece into a vector of numbers (meaning fingerprint)
  4. store             : save vectors + metadata (file name, page) in ChromaDB
  5. retrieve()        : embed the question, find the most similar pieces
  6. answer()          : give those pieces + the question to an LLM, get a cited answer
"""

import os
import re

import chromadb
import fitz  # PyMuPDF - reads PDFs
from groq import Groq
from sentence_transformers import SentenceTransformer

# ----------------------------- settings ------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, free, runs on a normal laptop CPU
LLM_MODEL = "openai/gpt-oss-120b"  # free on Groq
CHUNK_SIZE = 800      # characters per chunk (~120-150 words)
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
DB_PATH = "chroma_db"  # folder where the vector database is saved


# ----------------------------- step 1: read PDF -----------------------------
def load_pdf_pages(file_bytes: bytes):
    """Return a list of (page_number, text). Page numbers start at 1."""
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                pages.append((i, text))
    return pages


# ----------------------------- step 2: chunking -----------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping windows, preferably ending at a space."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:  # don't cut a word in half
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 40]  # drop tiny junk pieces


# ----------------------------- the engine -----------------------------------
class RAGEngine:
    def __init__(self, api_key: str | None = None):
        # Embedding model: downloads ~90 MB the first time, then works offline.
        self.embedder = SentenceTransformer(EMBED_MODEL)
        # Vector database saved on disk, so uploads survive app restarts.
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="course_material",
            metadata={"hnsw:space": "cosine"},  # compare vectors by cosine similarity
        )
        self.llm = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    # ---- steps 1-4: ingestion ----
    def add_document(self, name: str, file_bytes: bytes) -> int:
        """Read, chunk, embed and store one PDF. Returns number of chunks."""
        self.delete_document(name)  # re-uploading replaces the old version
        ids, texts, metas = [], [], []
        for page_no, page_text in load_pdf_pages(file_bytes):
            for j, chunk in enumerate(chunk_text(page_text)):
                ids.append(f"{name}-p{page_no}-c{j}")
                texts.append(chunk)
                metas.append({"source": name, "page": page_no})
        if not texts:
            return 0  # probably a scanned PDF with no selectable text
        vectors = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
        return len(texts)

    def list_documents(self):
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def delete_document(self, name: str):
        self.collection.delete(where={"source": name})

    # ---- step 5: retrieval ----
    def retrieve(self, question: str, top_k: int = 5, sources: list[str] | None = None):
        """Return the top_k most similar chunks as dicts with a similarity score."""
        if self.collection.count() == 0:
            return []
        q_vec = self.embedder.encode([question], normalize_embeddings=True).tolist()
        where = {"source": {"$in": sources}} if sources else None
        res = self.collection.query(
            query_embeddings=q_vec,
            n_results=min(top_k, self.collection.count()),
            where=where,
        )
        hits = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "text": text,
                    "source": meta["source"],
                    "page": meta["page"],
                    "score": round(1 - dist, 3),  # cosine distance -> similarity
                }
            )
        return hits

    # ---- helper: make follow-up questions self-contained ----
    def condense_question(self, history: list[dict], question: str) -> str:
        """'Explain it more simply' means nothing alone; rewrite using chat history."""
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        prompt = (
            "Rewrite the student's last question so it is fully self-contained, "
            "using the conversation for context. Output ONLY the rewritten question.\n\n"
            f"Conversation:\n{convo}\n\nLast question: {question}"
        )
        out = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return out.choices[0].message.content.strip()

    # ---- step 6: generation ----
    def answer(self, question: str, hits: list[dict]) -> str:
        context = "\n\n".join(
            f"[{i}] (Source: {h['source']}, Page {h['page']})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        )
        system = (
            "You are a Course Doubt-Solver for students. Use ONLY the numbered "
            "context passages provided; never add facts, proofs or examples that "
            "are not in them. If the passages only partly answer the question, "
            "say what is missing. If they do not contain the answer, reply exactly: "
            "'I could not find this in your uploaded course material.' "
            "Cite every claim as (file name, p. PAGE), using the real file name "
            "and page shown in each passage header, never the passage number. "
            "Write all math in LaTeX between $ signs, like $x < n$. "
            "Be concise and clear."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"
        out = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,  # low = factual, less creative
        )
        text = out.choices[0].message.content.strip()
        # Streamlit only renders $...$ and $$...$$, so convert the other LaTeX style
        text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.S)
        text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.S)
        return text


            # ---- study tools: same retrieval, different instruction ----
    def study_tool(self, mode: str, topic: str, hits: list[dict]) -> str:
        """mode is 'summary' or 'quiz'. Builds study material from retrieved passages."""
        context = "\n\n".join(
            f"[{i}] (Source: {h['source']}, Page {h['page']})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        )
        tasks = {
            "summary": (
                "Write a clear study summary of the topic as bullet points: key "
                "definitions, main results or formulas, and a one-line takeaway."
            ),
            "quiz": (
                "Write 5 exam-style practice questions on the topic, from easy to "
                "harder. Then add an 'Answers' section with a short answer to each, "
                "citing where it comes from."
            ),
        }
        system = (
            "You are a study assistant for students. Use ONLY the numbered context "
            "passages provided; never add facts that are not in them. "
            "Cite sources as (file name, p. PAGE) using the real file name and page "
            "from each passage header, never the passage number. "
            "Write all math in LaTeX between $ signs, like $x < n$. "
            + tasks[mode]
        )
        user = f"Context:\n{context}\n\nTopic: {topic}"
        out = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        text = out.choices[0].message.content.strip()
        # Streamlit only renders $...$ and $$...$$, so convert the other LaTeX style
        text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.S)
        text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.S)
        return text
