from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, g

from mvp_rag.config import Settings
from mvp_rag.logging_utils import configure_logging
from mvp_rag.utils.chunk import chunk_text
from mvp_rag.utils.files import FileValidationError, delete_file, save_upload
from mvp_rag.utils.llm import LLMClient, build_context_block
from mvp_rag.utils.pdf import extract_text
from mvp_rag.utils.vectordb import VectorStore


# Set up a logger for this module. This is a standard Python practice.
LOGGER = logging.getLogger(__name__)
# Define a standard response to use when the AI cannot find an answer in the documents.
FALLBACK_ANSWER = "I don’t know from the documents."


# This function is the "application factory." It creates and configures the Flask app.
# Using a factory makes the app more modular and easier to test.
def create_app(testing: bool = False) -> Flask:
    # Load application settings from environment variables.
    # This keeps sensitive data like API keys out of the source code.
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["SETTINGS"] = settings
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.config["TESTING"] = testing

    # Make sure the directory for temporary file uploads exists.
    settings.ensure_upload_dir()

    # This function runs before every request.
    # We use it to assign a unique ID to each request for better logging and tracking.
    @app.before_request
    def assign_request_id() -> None:
        # `g` is a special Flask object that's available for the life of a single request.
        g.request_id = str(uuid4())

    # This function runs after every request.
    # It adds the unique request ID to the response headers.
    @app.after_request
    def add_request_id(response):
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    # This is a custom error handler for our specific `FileValidationError`.
    # It ensures that if a user uploads a bad file, they get a clear error message.
    @app.errorhandler(FileValidationError)
    def handle_file_error(exc: FileValidationError):
        LOGGER.warning("File validation error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 400

    # This is a general error handler for any other exceptions that might occur.
    # It prevents the app from crashing and provides a generic "Internal server error" message.
    @app.errorhandler(Exception)
    def handle_exception(exc: Exception):
        LOGGER.exception("Unhandled error: %s", exc)
        return jsonify({"ok": False, "error": "Internal server error."}), 500

    # Connect the URL routes (like "/" or "/ingest") to their corresponding Python functions.
    register_routes(app)
    return app


# This function contains all the URL endpoints for the application.
def register_routes(app: Flask) -> None:
    # The main route ("/") serves the single-page user interface.
    @app.get("/")
    def index():
        # `render_template` finds and renders the `index.html` file from the `templates` folder.
        return render_template("index.html")

    # The "/ingest" route handles the PDF file uploads. It only accepts POST requests.
    @app.post("/ingest")
    def ingest():
        # Basic validation to ensure files were actually sent with the request.
        if "files" not in request.files:
            return jsonify({"ok": False, "error": "No files part in request."}), 400

        files = request.files.getlist("files")
        if not files:
            return jsonify({"ok": False, "error": "Please upload at least one PDF."}), 400

        # Initialize the clients for the vector database (Qdrant) and the LLM (OpenAI).
        # This is wrapped in a try...except block to handle potential configuration errors.
        try:
            vector_store = get_vector_store(app)
            llm_client = get_llm_client(app)
        except RuntimeError as exc:
            LOGGER.error("Ingest blocked: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

        settings: Settings = app.config["SETTINGS"]
        documents: List[Dict[str, Any]] = []  # To store results for each uploaded document.
        total_ingested = 0
        total_skipped = 0

        # Process each uploaded file one by one.
        for file_storage in files:
            destination = None  # Path where the temporary file will be saved.
            try:
                # 1. Save the uploaded file to a temporary location on the server.
                destination, file_hash = save_upload(
                    file_storage, settings.upload_dir, settings.max_upload_bytes
                )
                doc_title = destination.stem  # Use the filename (without extension) as the document title.

                # 2. Extract text from the PDF.
                pages = extract_text(destination)
                # 3. Split the extracted text into smaller, manageable chunks.
                chunks = chunk_text(pages)
                if not chunks:
                    LOGGER.info(
                        "No extractable text in %s (hash=%s)", destination.name, file_hash
                    )
                    documents.append(
                        {
                            "doc_title": doc_title,
                            "chunks_ingested": 0,
                            "chunks_skipped": 0,
                            "hash": file_hash,
                        }
                    )
                    continue

                # 4. Convert the text chunks into numerical vectors (embeddings) using the OpenAI API.
                embeddings = llm_client.embed_texts([chunk["chunk_text"] for chunk in chunks])
                # 5. Store these embeddings in the Qdrant vector database.
                ingested, skipped = vector_store.upsert_chunks(doc_title, chunks, embeddings)

                total_ingested += ingested
                total_skipped += skipped
                documents.append(
                    {
                        "doc_title": doc_title,
                        "chunks_ingested": ingested,
                        "chunks_skipped": skipped,
                        "hash": file_hash,
                    }
                )
                LOGGER.info(
                    "Ingested document %s (hash=%s) chunks=%s skipped=%s",
                    destination.name,
                    file_hash,
                    ingested,
                    skipped,
                )
            finally:
                # 6. Clean up by deleting the temporary file after it has been processed.
                if destination:
                    delete_file(destination)

        # Return a JSON response summarizing the ingestion process.
        return jsonify(
            {
                "ok": True,
                "documents": documents,
                "chunks_added": total_ingested,
                "chunks_skipped": total_skipped,
            }
        )

    # The "/ask" route handles user questions. It only accepts POST requests.
    @app.post("/ask")
    def ask():
        # Extract the user's question from the JSON payload of the request.
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"ok": False, "error": "Question is required."}), 400

        # Initialize the vector store and LLM clients.
        try:
            vector_store = get_vector_store(app)
            llm_client = get_llm_client(app)
        except RuntimeError as exc:
            LOGGER.error("Ask blocked: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

        # 1. Convert the user's question into an embedding.
        query_embedding = llm_client.embed_query(question)
        # 2. Use this embedding to search for the most relevant text chunks in the vector store.
        results = vector_store.query(query_embedding)

        # 3. Assemble the retrieved chunks into a "context" block to send to the LLM.
        context_blocks: List[str] = []
        citations: List[Dict[str, str]] = []  # Keep track of sources for citations.
        for point in results:
            payload = point.payload or {}
            doc_title = payload.get("doc_title")
            page_start = payload.get("page_start")
            page_end = payload.get("page_end")
            chunk_text = payload.get("chunk_text")
            if not (doc_title and page_start is not None and page_end is not None and chunk_text):
                continue
            context_blocks.append(build_context_block(doc_title, page_start, page_end, chunk_text))
            citations.append(
                {"doc": doc_title, "page_range": f"{page_start}-{page_end}"}
            )

        # 4. Send the context and the original question to the LLM to get a final answer.
        answer = llm_client.ask_llm(context_blocks, question)
        truncated_citations = citations[:4]  # Limit to the top 4 most relevant citations.

        # If the LLM couldn't find an answer, don't show any citations.
        if answer == FALLBACK_ANSWER:
            truncated_citations = []

        # 5. Return the answer and citations to the frontend.
        return jsonify({"ok": True, "answer": answer, "citations": truncated_citations})


# The `@lru_cache` decorator "memoizes" the function's result.
# This means the VectorStore client is created only once and then reused for all subsequent calls,
# which is much more efficient than creating a new connection every time.
@lru_cache(maxsize=1)
def get_vector_store(app: Flask) -> VectorStore:
    settings: Settings = app.config["SETTINGS"]
    return VectorStore.from_settings(settings)


# This function is also cached for the same efficiency reasons.
# It creates the OpenAI client, which will be reused across requests.
@lru_cache(maxsize=1)
def get_llm_client(app: Flask) -> LLMClient:
    settings: Settings = app.config["SETTINGS"]
    return LLMClient.from_settings(settings)


# This block of code runs only when the script is executed directly (e.g., `python -m mvp_rag.app`).
# It's the standard way to start a development server in Flask.
if __name__ == "__main__":
    flask_app = create_app()
    # Use the PORT environment variable if it's set, otherwise default to 8000.
    port = int(os.environ.get("PORT", "8000"))
    # Run the Flask development server.
    # host="0.0.0.0" makes it accessible from other devices on the same network.
    flask_app.run(host="0.0.0.0", port=port, debug=False)
