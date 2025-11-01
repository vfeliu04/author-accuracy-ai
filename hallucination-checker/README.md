# To Do:

- Fix the PDF reader. If the PDFs vary (financial reports, research papers, product docs, scans, etc.), the trick is to route each file to the right extractor and keep a generic fallback when there’s no perfect specialist. You don’t need to build heavy ML to get good results.
- Fix the table output, at the moment the start of the evidence is for some reason showing as text and the rest as table. 

- Still a major problem when reading the data from the PDF, this is something which is maybe disrupting the algorithm, solving this would help overall. 


# Hallucination Checker

A lightweight verification pipeline that compares factual claims made in a PDF report against supporting source PDFs. The project ships both a command-line interface (CLI) for scripted workflows and a Flask web interface for interactive exploration of results.

## Capabilities
- Extract narrative text, metadata, and table placeholders from source PDFs using Unstructured with PyMuPDF fallbacks, then store chunks in SQLite.
- Embed those chunks with OpenAI embeddings and persist a FAISS similarity index for quick semantic lookups.
- Detect numeric claims in a target report, retrieve and optionally GPT-rerank supporting snippets, and label each claim as `SUPPORTED`, `CONTRADICTED`, or `NOT_FOUND`.
- Serve the same verification flow through a web dashboard that handles file uploads, claim browsing, and explanation regeneration.

## Prerequisites
- Python 3.10+
- `git`
- Conda or `python -m venv` (or another virtual environment tool)
- An OpenAI API key (billing applies; monitor your usage)
- Native dependencies for [`unstructured`](https://github.com/Unstructured-IO/unstructured) such as `libmagic` and `poppler` if they are not already on your system

## Quick Start
1. **Clone the repository**
   ```bash
   git clone <your-fork-or-url> hallucination-checker
   cd hallucination-checker
   ```
2. **Create and activate a virtual environment**
   - Conda
     ```bash
     conda create -n hallcheck python=3.10 -y
     conda activate hallcheck
     ```
   - `venv`
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate  # Windows: .venv\Scripts\activate
     ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the NLTK Punkt tokenizer (once per environment)**
   ```bash
   python -m nltk.downloader punkt
   ```
5. **Configure environment variables**
   Create a `.env` file in the project root (or edit the existing one) and set at least:
   ```ini
   OPENAI_API_KEY=sk-your-key
   HALLCHECK_DB_URL=sqlite:///./hallcheck.db
   OPENAI_EMBED_MODEL=text-embedding-3-large
   OPENAI_CHAT_MODEL=gpt-4o-mini
   ```
   Adjust other settings as needed (see [Tunable Settings](#tunable-settings)).

## Directory Layout
- `data/inputs/sources/` – source PDFs that contain ground-truth evidence
- `data/inputs/reports/` – report PDFs whose claims you want to verify
- `data/indexes/` – FAISS index files and metadata generated during `index`
- `hallcheck.db` – SQLite database storing documents, chunks, claims, verdicts

Both the CLI and web UI read and write to these directories. The web app can also upload PDFs directly into the `data/inputs/...` folders.

## Running the CLI
1. Populate your sources and reports directories (or point to files elsewhere).
2. **Index sources** – this resets the database and index artifacts for the given index name:
   ```bash
   python -m src.hallcheck.cli index --sources data/inputs/sources/source1.pdf data/inputs/sources/source2.pdf
   ```
3. **Verify a report** – run against the previously indexed sources:
   ```bash
   python -m src.hallcheck.cli verify --report data/inputs/reports/report.pdf
   ```
   The CLI prints a summary per claim, including the best evidence snippet and verdict. Each new `index` run wipes stored results for that index to keep data consistent with the currently registered sources.

## Running the Web UI
1. Ensure your `.env` is configured and dependencies are installed.
2. Place PDFs in the `data/inputs/...` folders **or** upload them through the UI.
3. Start the Flask app:
   ```bash
   python app.py
   # or
   FLASK_APP=app.py flask run
   ```
4. Visit http://127.0.0.1:5000 to:
   - Upload or select source PDFs and trigger an indexing run.
   - Upload or select a report for verification.
   - Explore verdicts, evidence snippets, tables, and cached explanations.
   - Regenerate explanations for individual claims when needed.

The CLI (`python -m src.hallcheck.cli ...`) remains fully supported alongside the web experience; both share the same database and index.

## Claim Workflow Highlights
- Verification extracts numeric claims heuristically, embeds them, and retrieves the top `topk` (default 5) evidence chunks from FAISS.
- If `RERANK_WITH_GPT` is enabled, the rerank model (`gpt-4o-mini` by default) rescoring determines the top candidate.
- Verdict heuristics weigh numeric alignment, year consistency, entity overlap, and (optionally) GPT-rendered relevance to assign `SUPPORTED`, `CONTRADICTED`, or `NOT_FOUND`.
- Explanations are generated with `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`) and cached in the database; regenerating them keeps the stored verdict but replaces the explanation text.

## Tunable Settings
All settings can be controlled via environment variables (defaults shown):

| Setting | Default | Purpose |
| --- | --- | --- |
| `CHUNK_TOKENS` | `220` | Target token count for adaptive chunking. |
| `CHUNK_OVERLAP_MIN` / `CHUNK_OVERLAP_MAX` | `20` / `80` | Dynamic overlap bounds between adjacent chunks. |
| `CHUNK_TOPIC_SIMILARITY` | `0.72` | Similarity threshold that signals a topic change. |
| `CHUNK_STABLE_SIMILARITY` | `0.86` | Similarity level considered stable enough for minimal overlap. |
| `RERANK_WITH_GPT` | `true` | Enables GPT reranking of FAISS candidates. |
| `RERANK_MODEL` | `gpt-4o-mini` | Model used for reranking and entity alignment checks. |
| `RERANK_MAX_CANDIDATES` | `5` | Number of FAISS hits reranked per claim. |

Use `OPENAI_CHAT_MODEL` or `RERANK_MODEL` values compatible with your installed `openai` SDK (`>=1.2.0` for `gpt-4.1` support).

## Inspecting Stored Results
```bash
sqlite3 hallcheck.db "SELECT id, sentence FROM claims LIMIT 5;"
```

```python
import json, sqlite3
conn = sqlite3.connect("hallcheck.db")
query = """
SELECT c.sentence, v.status, json_extract(v.evidence, '$.candidates[0].text')
FROM claims c
JOIN verdicts v ON c.id = v.claim_id
LIMIT 5;
"""
for sentence, status, evidence in conn.execute(query):
    print(status, sentence, evidence or "")
```

FAISS metadata lives in `data/indexes/<index_name>.faiss`, `<index_name>_docids.npy`, and `<index_name>_chunks.jsonl`.

## Troubleshooting
- **OpenAI authentication** – confirm `OPENAI_API_KEY` is present in `.env` or your shell (`export OPENAI_API_KEY=...` on Unix, `set` on Windows).
- **FAISS not found** – ensure you are using the environment where `faiss-cpu` was installed (on Apple Silicon, install the CPU wheel).
- **NLTK tokenizer errors** – re-run `python -m nltk.downloader punkt` inside the active environment.
- **PyMuPDF build issues** – install Xcode CLT on macOS, upgrade `pip` on Windows, or ensure `build-essential`/`python3-dev` on Linux.

## Ideas & Next Steps
1. Replace heuristic claim extraction with an LLM that returns structured claim metadata.
2. Refine numeric and unit comparison tolerances with fuzzy string matching for entities.
3. Produce downloadable HTML/Markdown reports summarizing verdicts and evidence.
