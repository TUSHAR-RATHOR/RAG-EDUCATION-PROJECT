# Course Doubt-Solver: How and Why (your briefing notes)

## 1. The problem in one line
Students lose hours hunting through PDFs. Generic chatbots (ChatGPT etc.) answer from the internet, not from *your syllabus*, and cite nothing. We built a chatbot that answers **only from the student's own uploaded material** and shows **document name + page number** for every answer.

## 2. What is RAG? (say this first)
**RAG = Retrieval-Augmented Generation.**
An LLM (like ChatGPT/Llama) only knows what it was trained on, and it can make things up ("hallucinate"). RAG fixes this with an open-book exam approach:
1. **Retrieve**: search the student's notes for the passages relevant to the question.
2. **Augment**: paste those passages into the prompt.
3. **Generate**: tell the LLM "answer using only these passages, and cite them."

Why not just paste the whole textbook into the LLM? It is too long for the model's input limit, slow, costly, and the model gets distracted. Retrieval picks the 5 best passages instead.

## 3. Pipeline and the reason behind each step

| # | Step | What it does | Why we do it | Tool |
|---|------|--------------|--------------|------|
| 1 | **Load PDF page by page** | Extract text from each page separately | So every piece of text remembers its **page number**, which is needed for citations | PyMuPDF |
| 2 | **Chunking** | Cut each page into ~800-character pieces with 150 characters of overlap | LLMs and search work better on small focused pieces. Overlap stops a sentence being cut in half at the boundary and losing meaning | our `chunk_text()` |
| 3 | **Embeddings** | Convert each chunk into a list of 384 numbers (a vector) | Computers can't compare *meaning* of text directly. Embeddings place text with similar meaning close together ("car" and "automobile" get similar vectors even with no shared words). This is far better than keyword search | `all-MiniLM-L6-v2` (free, runs locally) |
| 4 | **Vector database** | Store vectors + metadata (file, page) | Lets us quickly find the nearest vectors, and saves to disk so uploads persist | ChromaDB |
| 5 | **Retrieval** | Embed the question the same way, return the top-k closest chunks using cosine similarity | Finds the passages most relevant to the doubt. Cosine similarity measures the angle between two vectors: 1 = same meaning, 0 = unrelated | ChromaDB query |
| 6 | **Generation** | Send the chunks + question to the LLM with strict instructions | The LLM turns raw passages into a clear explanation. Rules in the prompt: use only the context, cite sources, say "not found" otherwise | GPT-OSS 120B |
| 7 | **UI** | Upload, chat, show sources | Students need something usable, not code | Streamlit |

## 4. Extra features (these earn marks, mention them)
- **Source references**: file name + page number on every answer, plus an expandable box showing the exact passages and their relevance score (transparency).
- **Hallucination guard**: if the best match's similarity is below a threshold, the app says "not found in your material" without even calling the LLM. The strict system prompt does the same.
- **Follow-up questions**: "explain it simpler" is meaningless alone, so `condense_question()` rewrites it into a standalone question using chat history before searching.
- **Multiple documents + filter**: search all notes or only selected files.
- **Adjustable settings**: top-k and minimum relevance sliders.
- **Persistent storage + delete**: documents stay indexed after restart and can be removed.

## 5. Key terms in plain words (viva-ready)
- **LLM**: a model trained on huge text that predicts and writes text.
- **Hallucination**: LLM confidently stating false things. RAG reduces it by grounding answers in documents.
- **Embedding**: numbers representing the meaning of text.
- **Vector database**: database optimised to find the nearest vectors.
- **Cosine similarity**: score of how similar two vectors' directions are.
- **Chunk / overlap**: small piece of text / shared text between neighbouring pieces.
- **top-k**: how many passages we retrieve.
- **Temperature**: LLM randomness. We use 0.2 (low) so answers stay factual.
- **Prompt / system prompt**: the instructions given to the LLM; the system prompt sets its rules.

## 6. Likely questions and answers
**Why not fine-tune a model on the notes?** Fine-tuning is expensive, needs lots of data and GPUs, can still hallucinate, and gives no page citations. RAG works instantly with any new PDF.

**Why chunk size 800 and overlap 150?** Big enough to hold a full idea, small enough to be precise. It is a tunable setting; too large gives noisy retrieval, too small loses context.

**Why a local embedding model?** Free, private (notes never leave the machine for the search step), no API limits.

**How do you stop hallucination?** Three layers: retrieval-only context, a strict prompt with a mandatory "not found" reply, and a similarity threshold.

**How do you get the page number?** We extract text page by page and store `{source, page}` as metadata on every chunk; retrieval returns that metadata with the text.

**Limitations?** Scanned/image PDFs need OCR (not included); tables and equations extract poorly; free LLM API has rate limits.

**Future work:** OCR for scanned notes, re-ranking retrieved chunks, support for PPT/DOCX, quiz generation, highlighting the passage inside the PDF, evaluation with a question-answer test set.

## 7. Suggested timeline for your one day
1. **(30 min)** Install and get it running with the steps in README.md.
2. **(1 hour)** Read `rag.py` top to bottom. Every step has comments, and it's only ~150 lines. Match it to the table in section 3.
3. **(1 hour)** Test with 2-3 real PDFs from your course. Write down 10 questions and check the answers and page numbers are right.
4. **(1 hour)** Experiment: change `CHUNK_SIZE`, top-k, and see how answers change. Note the results; examiners love this.
5. **(1 hour)** Prepare a 5-minute demo: upload, ask a normal question, ask a follow-up, ask something not in the notes (shows "not found"), open the sources box.
6. **(rest)** Rehearse sections 2, 3 and 6 out loud.

## 8. Demo script
1. State the problem (section 1) and what RAG is (section 2).
2. Upload a PDF and click Process; explain steps 1 to 4 happen now.
3. Ask a question; show the answer, the page citations, and the sources box (steps 5 and 6).
4. Ask a follow-up like "explain that with an example".
5. Ask something unrelated; show it refuses honestly.
6. Close with limitations and future work.
