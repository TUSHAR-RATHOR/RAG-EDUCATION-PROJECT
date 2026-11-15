# Course Doubt-Solver (RAG)

Upload course PDFs, ask questions, get answers grounded in your material with document name and page number.

## Setup (about 10 minutes)
1. Install Python 3.10+ (python.org). Check with `python --version`.
2. Put these files in one folder, open a terminal there.
3. (Recommended) create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install packages: `pip install -r requirements.txt` (takes a few minutes; sentence-transformers is large).
5. Get a free API key at https://console.groq.com (Sign up, API Keys, Create).
6. Copy `.env.example` to `.env` and paste your key (or paste it in the app's sidebar).
7. Run: `streamlit run app.py` and open the link it prints (http://localhost:8501).

The first run downloads the embedding model (~90 MB). After that it works fast.

## Use
Upload PDFs in the sidebar, click **Process documents**, then ask questions in the chat.

## Files
- `rag.py`: the RAG engine (read this to understand the project)
- `app.py`: Streamlit interface
- `EXPLAINED.md`: how and why of every step, viva questions, demo script

## Troubleshooting
- *Model decommissioned error*: Groq retires models sometimes. Set `LLM_MODEL` in `.env` to a current model from console.groq.com/docs/models.
- *"no text found"*: the PDF is scanned images; use a text-based PDF.
- To reset everything, delete the `chroma_db` folder.
