# MVP Retrieval-Augmented Generation App

A minimal Flask-based Retrieval-Augmented Generation (RAG) web application that ingests PDF documents, stores embeddings in Qdrant, and answers user questions with grounded citations.

## Features
- Upload 1–3 PDF files and ingest them into a single Qdrant collection.
- Chunking tuned for `text-embedding-3-small` (≈900 tokens, 100-token overlap).
- OpenAI `gpt-4o-mini` responses with strict “I don’t know from the documents.” fallback.
- One-page UI with upload zone, ingest status, question box, and citation results.
- Structured JSON logging with request IDs.

## Quickstart

1. **Clone and enter the project directory**, then create a conda environment:
   ```bash
   conda create -y -n mvp-rag python=3.11
   conda activate mvp-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit the file to add your keys
   ```

   Required variables:
   - `OPENAI_API_KEY` – OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`
   - `QDRANT_URL` – Qdrant endpoint (e.g., `http://localhost:6333`)
   - `QDRANT_API_KEY` – Qdrant API key (leave blank for local deployments)
   - Optional: `COLLECTION_NAME`, `UPLOAD_DIR`, `LOG_LEVEL`, `MAX_UPLOAD_MB`, `REQUEST_TIMEOUT_SECONDS`

4. **Run the development server**:
   ```bash
   python -m mvp_rag.app
   ```

   The app listens on `http://localhost:8000` by default. Adjust the `PORT` environment variable to override.

5. **Run tests**:
   ```bash
   pytest
   ```

   Tests stub external services so they succeed without API keys.

## Project Structure

```
mvp_rag/
  app.py            # Flask application factory and routes
  config.py         # Environment-driven settings
  logging_utils.py  # Structured logging configuration
  utils/            # PDF parsing, chunking, vector DB, LLM helpers, token utilities
  templates/        # Jinja template for the single-page UI
  static/           # CSS and JS assets
tests/              # Smoke tests with in-memory stubs and fixtures
uploads/            # Temporary upload directory (ignored from VCS)
```

## Operational Notes
- Temporary uploads are saved to `UPLOAD_DIR` and removed after ingestion.
- Logging excludes document contents and includes file hashes plus chunk counts.
- Qdrant collection schema stores chunk metadata with deduplication on `(doc_title, page_start, chunk_id)`.

## How It Works (for Complete Beginners)

### What is a RAG App?

A Retrieval-Augmented Generation (RAG) app is a smart system that can answer questions based on a specific set of documents you provide. Instead of using a generic AI model's vast (but sometimes outdated or irrelevant) knowledge, a RAG app does two things:

1.  **Retrieval:** It searches through your documents to find the exact pieces of text that are most relevant to your question.
2.  **Generation:** It takes those relevant text snippets and uses a powerful AI (like GPT) to generate a natural, human-like answer based *only* on that information.

Think of it like an open-book exam for an AI. It doesn't guess the answer; it looks it up in the book you gave it and then explains it to you.

### How This App Works: Step-by-Step

This application is a simple, self-contained RAG system. Here’s the journey from a PDF file to an AI-generated answer:

1.  **PDF Upload:** You start by uploading one or more PDF files through the web interface. The files are temporarily saved on the server.
2.  **Text Extraction:** The app opens each PDF and extracts all the text from its pages.
3.  **Chunking:** The extracted text is broken down into smaller, overlapping "chunks" of about 900 tokens each. This is crucial because AI models have a limited context window (how much text they can read at once), and it helps pinpoint specific information.
4.  **Embedding:** Each text chunk is converted into a numerical representation called a "vector embedding" using an OpenAI model. These embeddings capture the semantic meaning of the text, allowing for "concept-based" searching rather than just keyword matching.
5.  **Storing in Qdrant:** The embeddings and their corresponding text chunks (with metadata like the document title and page number) are stored in a specialized database called a **vector database** (in this case, Qdrant).
6.  **Asking a Question:** When you type a question, the app converts your question into an embedding as well.
7.  **Searching (Retrieval):** It then uses this question embedding to search the Qdrant database for the text chunks with the most similar embeddings. These are the chunks most relevant to your question.
8.  **Answering (Generation):** The app takes the top few relevant chunks, combines them into a "context," and sends them—along with your original question—to an OpenAI chat model. It instructs the model to answer the question strictly based on the provided context.
9.  **Displaying the Result:** The final answer, along with citations showing which document and page range the information came from, is displayed on the web page.

### Main Tools and Their Roles

-   **Flask:** A lightweight Python web framework used to build the web server and handle user requests.
-   **OpenAI API:** Provides the AI models for creating text embeddings (`text-embedding-3-small`) and generating answers (`gpt-4o-mini`).
-   **Qdrant:** The vector database that stores the embeddings and allows for efficient similarity searches.
-   **pypdf:** A Python library used to read and extract text from PDF files.
-   **tiktoken:** A Python library from OpenAI used to count "tokens," which helps in correctly sizing the text chunks.

### Example Workflow

1.  **Upload:** Drag and drop a PDF file (e.g., a project report) into the "Document Ingestion" panel and click "Ingest."
2.  **Wait:** The app will show a status message as it processes the file. Once complete, it will confirm the number of text chunks added.
3.  **Ask:** In the "Ask the Documents" panel, type a question related to the report, like "What were the key findings in Q3?" and click "Ask."
4.  **Review:** The app will display an answer generated from the report's content, along with citations pointing to the specific pages where the information was found.
