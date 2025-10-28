# MVP RAG Assistant User Manual

This guide walks you through installing, configuring, and operating the MVP Retrieval-Augmented Generation (RAG) Flask application. The app lets you upload PDFs, builds a Qdrant vector index with OpenAI embeddings, and answers questions with citations pulled directly from your documents.

## System Requirements & Prerequisites
- Python 3.11 or newer
- Conda (Miniconda or Anaconda) with `conda` available in your shell
- Git 2.30+ for cloning and version control
- A Qdrant Cloud account (free tier is sufficient)
- An OpenAI API key with access to `text-embedding-3-small` and GPT response models
- macOS/Linux terminal or Windows PowerShell (commands for both are included)

## Quick Start

### macOS / Linux
```bash
git clone https://github.com/your-org/author_-accuracy.ai.git
cd author_-accuracy.ai
conda create -n mvp_rag python=3.11 -y
conda activate mvp_rag
pip install -r requirements.txt
cp .env.example .env
python -m mvp_rag.app
```

### Windows PowerShell
```powershell
git clone https://github.com/your-org/author_-accuracy.ai.git
Set-Location author_-accuracy.ai
conda create -n mvp_rag python=3.11 -y
conda activate mvp_rag
pip install -r requirements.txt
Copy-Item .env.example .env
python -m mvp_rag.app
```

## Detailed Setup

### 1. Clone the repository
- **macOS/Linux**
  ```bash
  git clone https://github.com/your-org/author_-accuracy.ai.git
  cd author_-accuracy.ai
  ```
- **Windows PowerShell**
  ```powershell
  git clone https://github.com/your-org/author_-accuracy.ai.git
  Set-Location author_-accuracy.ai
  ```

### 2. Create & activate the Conda environment
- **macOS/Linux**
  ```bash
  conda create -n mvp_rag python=3.11 -y
  conda activate mvp_rag
  ```
- **Windows PowerShell**
  ```powershell
  conda create -n mvp_rag python=3.11 -y
  conda activate mvp_rag
  ```

### 3. Install Python dependencies
- **macOS/Linux**
  ```bash
  pip install -r requirements.txt
  ```
- **Windows PowerShell**
  ```powershell
  pip install -r requirements.txt
  ```

### 4. Configure environment variables
1. Duplicate the sample environment file:
   - **macOS/Linux**
     ```bash
     cp .env.example .env
     ```
   - **Windows PowerShell**
     ```powershell
     Copy-Item .env.example .env
     ```
2. Open `.env` in your editor and replace placeholder values. A complete example:
   ```dotenv
   OPENAI_API_KEY=sk-your-openai-key
   QDRANT_URL=https://your-cluster-xxxxxx.us-east-1-0.aws.cloud.qdrant.io
   QDRANT_API_KEY=your-qdrant-api-key
   COLLECTION_NAME=mvp_docs
   PORT=8000
   LOG_LEVEL=INFO
   UPLOAD_DIR=uploads
   MAX_UPLOAD_MB=20
   REQUEST_TIMEOUT_SECONDS=30
   ```

### 5. Qdrant Cloud setup
1. Visit <https://cloud.qdrant.io/> and **Log in** (create a free account if needed).
2. Click **Create cluster**, choose the smallest available configuration, name it (for example, `mvp-rag`), and confirm.
3. After provisioning, open your cluster and copy the **REST endpoint URL** (used as `QDRANT_URL`).
4. Go to **Access Management → API Keys**, click **Create API Key**, give it a name (e.g., `rag-app`), enable **Read** and **Write**, and copy the generated key for `QDRANT_API_KEY`.
5. In the cluster view, click **Collections → Create collection**. Name it `mvp_docs`, keep the default size/distance (the app expects size 1536 and cosine distance), and submit. If you skip this step, the app will attempt to create it automatically on first ingest.
6. Update `.env` with the endpoint URL and API key you just created.

## Running the App Locally
1. Ensure your Conda environment is active (`conda activate mvp_rag`).
2. Start the Flask server:
   - **macOS/Linux**
     ```bash
     python -m mvp_rag.app
     ```
   - **Windows PowerShell**
     ```powershell
     python -m mvp_rag.app
     ```
3. By default the app listens on `http://localhost:8000`. To run on another port:
   - **macOS/Linux**
     ```bash
     export PORT=5000
     python -m mvp_rag.app
     ```
   - **Windows PowerShell** (temporary for current session)
     ```powershell
     $Env:PORT = "5000"
     python -m mvp_rag.app
     ```

Temporary PDF uploads are stored in the `uploads/` directory and are removed after ingestion.

## Using the App
1. Open your browser at `http://localhost:8000`.
2. Click **Upload PDFs** or drag-and-drop one or more PDF files.
3. Press **Ingest**. Wait for the success message showing ingested and skipped chunk counts.
4. Type a question into the **Ask** input and click **Submit**.
5. Read the AI response and review the citation cards showing document titles and page ranges.
6. Repeat ingestion when you add or update documents; previously ingested chunks are deduplicated.

## API Reference

### POST `/ingest`
- **Description:** Upload one or more PDFs for ingestion.
- **Request:** `multipart/form-data` with one or more `files` fields (`Content-Type: application/pdf`).
- **Response (200):**
  ```json
  {
    "ok": true,
    "documents": [
      {
        "doc_title": "example",
        "chunks_ingested": 12,
        "chunks_skipped": 0,
        "hash": "…"
      }
    ],
    "chunks_added": 12,
    "chunks_skipped": 0
  }
  ```

### POST `/ask`
- **Description:** Retrieve an answer based on ingested content.
- **Request body (`application/json`):**
  ```json
  { "question": "What does the architecture look like?" }
  ```
- **Response (200):**
  ```json
  {
    "ok": true,
    "answer": "Detailed answer with citations.",
    "citations": [
      { "doc": "example", "page_range": "2-3" }
    ]
  }
  ```

### cURL examples
- **Ingest (macOS/Linux):**
  ```bash
  curl -X POST \
    -F "files=@/absolute/path/to/file.pdf" \
    http://localhost:8000/ingest
  ```
- **Ingest (Windows PowerShell):**
  ```powershell
  curl.exe -X POST `
    -F "files=@C:\path\to\file.pdf" `
    http://localhost:8000/ingest
  ```
- **Ask (macOS/Linux):**
  ```bash
  curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"question":"What is covered in section 2?"}' \
    http://localhost:8000/ask
  ```
- **Ask (Windows PowerShell):**
  ```powershell
  curl.exe -X POST `
    -H "Content-Type: application/json" `
    -d '{"question":"What is covered in section 2?"}' `
    http://localhost:8000/ask
  ```

## Testing & Validation
1. Run the automated test suite:
   - **macOS/Linux**
     ```bash
     pytest -q
     ```
   - **Windows PowerShell**
     ```powershell
     pytest -q
     ```
2. Tests that require live OpenAI or Qdrant access are automatically skipped when `OPENAI_API_KEY` or `QDRANT_API_KEY` is missing. Set the keys in your environment to run the full suite.
3. Validation checklist before release:
   - [ ] All tests pass (`pytest -q`)
   - [ ] `.env` contains valid OpenAI and Qdrant credentials
   - [ ] Ingesting a sample PDF succeeds and reports chunk counts
   - [ ] Asking a question returns citations tied to uploaded documents
   - [ ] Deployment environment variables match production secrets

## Troubleshooting Guide

| Issue | Likely Cause | Fix |
| --- | --- | --- |
| `401 Unauthorized` from Qdrant | `QDRANT_API_KEY` or `QDRANT_URL` incorrect | Regenerate the API key in Qdrant Cloud and update `.env`; ensure the URL matches the REST endpoint. |
| OpenAI authentication error | Missing or invalid `OPENAI_API_KEY` | Confirm the key is copied without whitespace; export it before launching the app. |
| “No documents ingested” message | PDFs not successfully processed | Verify the upload succeeded and `chunks_added` > 0; re-upload and click **Ingest** again. |
| “I don’t know from the documents.” answer | Question not covered in retrieved context or no documents available | Ingest relevant PDFs, rephrase the question, or ensure ingestion succeeded. |
| Large PDF rejected | File exceeds `MAX_UPLOAD_MB` (default 20 MB) | Split the PDF, raise `MAX_UPLOAD_MB` in `.env`, or compress the document. |
| CORS error in browser console | Accessing the API from a different origin | Run the UI and API on the same origin or add a proxy in front of the Flask app that handles CORS. |

## Security & Privacy
- Sensitive keys live in `.env` and are not logged if `.env` is excluded from version control.
- Uploaded PDFs are stored temporarily in `uploads/` and removed after ingestion completes.
- The app does not log raw document content; logs focus on metadata (hashes, chunk counts).
- Secure your deployment with HTTPS, environment-specific API keys, and least-privilege Qdrant keys.

## Deployment

### Render
1. Connect your GitHub repository to Render and create a new **Web Service**.
2. Set the **Build Command:** `pip install -r requirements.txt`
3. Set the **Start Command:** `python -m mvp_rag.app`
4. Add environment variables under **Environment → Add Secret**:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `COLLECTION_NAME` (use `mvp_docs`)
   - `PORT` (Render sets `$PORT`; leave blank to inherit)
   - Optional tuning: `LOG_LEVEL`, `MAX_UPLOAD_MB`, `REQUEST_TIMEOUT_SECONDS`
5. Enable a health check path such as `/` (Render checks for HTTP 200).

### Railway
1. Create a new **Service → Deploy from Repo** and select this project.
2. In the **Variables** tab, add the same environment variables as above.
3. Configure the **Start Command:** `python -m mvp_rag.app`
4. Railway auto-installs dependencies if you keep `pip install -r requirements.txt` in the **Build** section or add it to the **Nixpacks** hook.
5. Expose the port by ensuring `PORT` is set to `8000` or leaving it to Railway’s assigned value and reading it via `PORT`.

## FAQ
- **Why does the app respond “I don’t know from the documents.”?**  
  Either no relevant chunks were retrieved or no documents were ingested. Upload applicable PDFs and re-ask the question with more context.
- **How large can my PDFs be?**  
  The default maximum is 20 MB. Increase `MAX_UPLOAD_MB` in `.env` if you need more headroom and your infrastructure can handle it.
- **How do I clean the vector store?**  
  Use the Qdrant Cloud UI: open your collection → **Actions → Delete points** and filter by document payload, or delete and recreate the collection to start fresh.

## Glossary
- **RAG (Retrieval-Augmented Generation):** An approach that combines document retrieval with generative AI to ground answers in specific sources.
- **Embedding:** A numerical vector representation of text used for similarity search.
- **Chunk:** A manageable slice of text extracted from a larger document for indexing.
- **top_k:** The number of most relevant chunks retrieved during a query (defaults to 4 in this app).
- **Citation:** Metadata pointing to the source document and page range used to generate an answer.

## Appendix: Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI key with embedding and chat access | _None_ (required) |
| `QDRANT_URL` | Qdrant Cloud REST endpoint | _None_ (required) |
| `QDRANT_API_KEY` | Qdrant API key with read/write scope | _None_ (required) |
| `COLLECTION_NAME` | Qdrant collection used for storage | `mvp_docs` |
| `PORT` | HTTP port for the Flask server | `8000` |
| `LOG_LEVEL` | Application logging verbosity | `INFO` |
| `UPLOAD_DIR` | Temporary directory for uploads | `uploads` |
| `MAX_UPLOAD_MB` | Maximum upload size in megabytes | `20` |
| `REQUEST_TIMEOUT_SECONDS` | Client timeout for external APIs | `30` |

Optional: export this manual to PDF with a Markdown tool such as `pypandoc`:
- **macOS/Linux**
  ```bash
  pip install pypandoc
  pypandoc docs/USER_MANUAL.md -o docs/USER_MANUAL.pdf
  ```
- **Windows PowerShell**
  ```powershell
  pip install pypandoc
  pypandoc docs/USER_MANUAL.md -o docs/USER_MANUAL.pdf
  ```
