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
- System packages required by [`unstructured`](https://github.com/Unstructured-IO/unstructured) (e.g., `libmagic`, `poppler`, and related native libraries) if they are not already installed on your machine.

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
1. Place your PDF sources inside `data/inputs/sources/` and reports inside `data/inputs/reports/`. During indexing the pipeline uses the `unstructured` library (hi_res strategy with fallbacks) and heuristics to strip header/footer/license blocks, so only main narrative text and key metadata go into the database and FAISS index.
2. Index source documents:
   ```bash
   python -m src.hallcheck.cli index --sources data/inputs/sources/source1.pdf data/inputs/sources/source2.pdf
   ```
3. Verify a report:
   ```bash
   python -m src.hallcheck.cli verify --report data/inputs/reports/report.pdf
   ```

Running `index` automatically clears the SQLite tables and any existing FAISS files for that index name, so each run starts from a clean slate. The CLI prints progress messages and, after verification, a summary showing each claim sentence, the highest-scoring source snippet, and whether the claim is SUPPORTED, CONTRADICTED, or NOT_FOUND. Re-run `index` whenever you change the set of source PDFs.

## Run the Web UI
1. Ensure your `.env` file is configured (set `OPENAI_API_KEY`, optionally tweak chunking/index settings).
2. Install dependencies, including the Flask extras: `pip install -r requirements.txt`.
3. Download the NLTK tokenizer if you have not already: `python -m nltk.downloader punkt`.
4. Place source PDFs inside `data/inputs/sources/` and report PDFs inside `data/inputs/reports/` ahead of time, or upload them directly from the UI (uploads are saved into those folders).
5. Start the server with either `FLASK_APP=app.py flask run` or `python app.py`.
6. Open http://127.0.0.1:5000 in your browser to index sources, verify a report, and browse results.

The original CLI (`python -m src.hallcheck.cli ...`) continues to work unchanged alongside the new web interface.

## Claim detail page
- From the verification results screen, click any claim card to open a dedicated detail view.
- The page shows the full claim sentence, the GPT-reranked evidence snippet (with expandable full context), and a cached AI explanation for the verdict.
- Explanations are grounded strictly in the stored claim and evidence context—the LLM is prompted to restate the claim, cite matching numbers/years directly from the snippet, and conclude the verdict without speculating beyond the provided text.
- Explanations are generated with `OPENAI_CHAT_MODEL` (defaults to `gpt-4.1`) and stored in the database; use the **Regenerate explanation** button to refresh them on demand.
- Placeholder screenshot: `![Claim detail screenshot](docs/claim-detail.png)`

## Hybrid retrieval & reranking
- Source PDFs are first segmented into topical blocks (TextTiling with fallbacks), then chunked with adaptive sentence windows so chunks stop at topic changes.
- Chunk overlap is dynamic: stable sections keep small overlaps, fast-changing sections keep more context, balancing recall and index size.
- After FAISS retrieval, candidate chunks are trimmed to the most relevant sentences and (optionally) reranked with `gpt-4o-mini` for a refined relevance score.
- The top reranked snippet feeds the verdict heuristics, cached explanations, and UI, yielding shorter yet accurate evidence.
- During ingestion we rely on the `unstructured` library to pull narrative text and metadata (title, authors, publication year); only the cleaned body text is chunked and indexed, cutting out headers, footers, and reference noise.

### Tunable settings
Configure the pipeline via environment variables:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CHUNK_TOKENS` | `220` | Target token budget per adaptive chunk. |
| `CHUNK_OVERLAP_MIN` / `CHUNK_OVERLAP_MAX` | `20` / `80` | Bounds for dynamic overlap when context shifts. |
| `CHUNK_TOPIC_SIMILARITY` | `0.72` | Similarity threshold that triggers a new chunk. |
| `CHUNK_STABLE_SIMILARITY` | `0.86` | Similarity treated as “stable” (allows minimum overlap). |
| `RERANK_WITH_GPT` | `true` | Toggle GPT-driven evidence reranking. |
| `RERANK_MODEL` | `gpt-4o-mini` | Model used for reranking (falls back to chat completions if Responses API unavailable). |
| `RERANK_MAX_CANDIDATES` | `5` | Number of FAISS candidates reranked per claim. |

If you're on an older `openai` Python SDK without the Responses API, keep `OPENAI_CHAT_MODEL` or `RERANK_MODEL` on `gpt-4o-mini`, or upgrade to `openai>=1.2.0` for `gpt-4.1` support.

## Under the Hood
The tool reads PDFs with PyMuPDF, segments them into topical sections, and builds adaptive sentence chunks that respect context boundaries. Each chunk is embedded with `text-embedding-3-large`, persisted in SQLite, and indexed with FAISS. During verification it extracts numeric claims, embeds them, retrieves the top candidates, trims them to the best matching sentences, optionally reranks them with GPT, and passes the highest-scoring snippet into lightweight heuristics (numeric matching, year alignment, similarity thresholds) to assign SUPPORTED/CONTRADICTED/NOT_FOUND verdicts.

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
