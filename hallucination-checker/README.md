# Hallucination Checker MVP
A minimal command-line tool that flags factual claims in a PDF report as supported, contradicted, or not found within a set of source PDFs.

## What This MVP Does
- Extracts text from PDFs, chunks the content, and stores it in SQLite.
- Builds a FAISS vector index using OpenAI embeddings for semantic search.
- Heuristically finds numeric claims in a report and labels them using retrieved evidence.

## What It Does Not Do
- Perform deep natural-language reasoning or guarantee correctness.
- Provide a graphical UI or produce formatted reports.
- Automatically download or parse data from the web.

## Prerequisites
- Python 3.10 or newer.
- `git` for cloning the repository.
- Conda or `python -m venv` for environment management.
- An OpenAI account and API key (usage incurs costs; monitor your spend).

## Setup
1. **Clone the repo**
   ```bash
   git clone <your-fork-or-url> hallucination-checker
   cd hallucination-checker
   ```
2. **Create and activate an environment**
   - Conda:
     ```bash
     conda create -n hallcheck python=3.10 -y
     conda activate hallcheck
     ```
   - venv:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
     ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK punkt tokenizer**
   ```bash
   python -m nltk.downloader punkt
   ```
5. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set `OPENAI_API_KEY` to your key. Adjust other settings if desired.

## Running the CLI
1. Place your PDFs inside `data/inputs/`.
2. Index source documents:
   ```bash
   python -m src.hallcheck.cli index --sources data/inputs/source1.pdf data/inputs/source2.pdf
   ```
3. Verify a report:
   ```bash
   python -m src.hallcheck.cli verify --report data/inputs/report.pdf
   ```

Running `index` automatically clears the SQLite tables and any existing FAISS files for that index name, so each run starts from a clean slate. The CLI prints progress messages and, after verification, a summary showing each claim sentence, the highest-scoring source snippet, and whether the claim is SUPPORTED, CONTRADICTED, or NOT_FOUND. Re-run `index` whenever you change the set of source PDFs.

## Run the Web UI
1. Ensure your `.env` file is configured (set `OPENAI_API_KEY`, optionally tweak chunking/index settings).
2. Install dependencies, including the Flask extras: `pip install -r requirements.txt`.
3. Download the NLTK tokenizer if you have not already: `python -m nltk.downloader punkt`.
4. Place PDFs inside `data/inputs/` ahead of time or upload them directly from the UI (uploads are saved in this folder).
5. Start the server with either `FLASK_APP=app.py flask run` or `python app.py`.
6. Open http://127.0.0.1:5000 in your browser to index sources, verify a report, and browse results.

The original CLI (`python -m src.hallcheck.cli ...`) continues to work unchanged alongside the new web interface.

## Under the Hood
The tool reads PDFs with PyMuPDF, splits source documents into overlapping sentence chunks, and stores them in SQLite alongside metadata. It embeds the chunks using `text-embedding-3-large`, stores vectors in a FAISS index, and writes aligned metadata files. During verification it extracts numeric statements from the report, embeds those sentences, retrieves the top-k similar source chunks, and applies lightweight heuristics (numeric matching, year detection, and similarity thresholds) to assign SUPPORTED/CONTRADICTED/NOT_FOUND verdicts.

## Where Results Live
- SQLite database: `hallcheck.db` (configurable via `.env`).
  - Tables: `documents`, `chunks`, `claims`, `verdicts`.
- Vector data: `data/indexes/` with `.faiss`, `_docids.npy`, and `_chunks.jsonl`.

Inspect the database with:
```bash
sqlite3 hallcheck.db "SELECT id, sentence FROM claims LIMIT 5;"
```
Or via Python:
```python
import sqlite3, json
conn = sqlite3.connect("hallcheck.db")
for row in conn.execute("SELECT c.sentence, v.status, v.evidence FROM claims c JOIN verdicts v ON c.id = v.claim_id LIMIT 5;"):
    print(row[0], row[1], json.loads(row[2])["candidates"][0]["text"])
```

## Troubleshooting
- **OpenAI authentication**: Confirm `OPENAI_API_KEY` is set and exported in your shell (`export OPENAI_API_KEY=...` or equivalent on Windows).
- **FAISS missing**: Ensure you're in the Python environment where `faiss-cpu` was installed; on Apple Silicon, verify you installed the CPU wheel and not GPU.
- **NLTK punkt error**: Re-run `python -m nltk.downloader punkt` inside your active environment.
- **PyMuPDF install issues**:
  - macOS: Install Xcode command-line tools (`xcode-select --install`) if missing.
  - Windows: Use the latest pip (`python -m pip install --upgrade pip`) before installing requirements.
  - Linux: Make sure `build-essential` and `python3-dev` are installed.

## Next Steps & Ideas
1. Replace heuristic claim extraction with an LLM returning structured JSON.
2. Tune numeric/unit matching tolerances and add fuzzy string checks for entities.
3. Generate an HTML or Markdown report summarizing verdicts with evidence excerpts.
