from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from openai import OpenAI

from mvp_rag.config import Settings
from mvp_rag.utils.tokens import batch, trim_to_token_limit


# Standard logger setup.
LOGGER = logging.getLogger(__name__)

# Define the specific OpenAI models we'll be using.
# Using constants makes it easy to update the model names in one place.
EMBEDDING_MODEL = "text-embedding-3-small"  # For creating vector embeddings from text.
CHAT_MODEL = "gpt-4o-mini"  # For generating answers based on context.

# This is the "system prompt" that guides the AI's behavior.
# It sets the rules for how the AI should respond, ensuring it relies only on the provided documents.
SYSTEM_PROMPT = (
    "You answer strictly using the provided context.\n"
    "If the answer is not in the context, reply exactly: \"I don’t know from the documents.\"\n"
    "Keep answers concise and cite sources in [Title, Pages] format."
)


# A client class to encapsulate all interactions with the OpenAI API.
@dataclass(slots=True)
class LLMClient:
    client: OpenAI  # The official OpenAI client instance.

    # A factory method to create an LLMClient instance from our application settings.
    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        # It's important to ensure the API key is configured before trying to create the client.
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        # Initialize the OpenAI client with the API key and a request timeout.
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_seconds)
        return cls(client=client)

    # This method converts a list of text strings into a list of vector embeddings.
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        # The API works best with batches of text, so we process the list in chunks.
        for chunk in batch(list(texts), size=64):
            # Call the OpenAI embeddings API.
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=list(chunk))
            # Extract the embedding vectors from the API response.
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    # This method is similar to `embed_texts` but is optimized for a single query string.
    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        return response.data[0].embedding

    # This is the core RAG function. It takes the retrieved context and the user's question
    # and asks the LLM to generate an answer.
    def ask_llm(self, context_blocks: Sequence[str], question: str) -> str:
        # If the vector search returned no relevant context, we can't answer the question.
        if not context_blocks:
            return "I don’t know from the documents."
        # Combine the context blocks into a single string to form the main prompt.
        prompt = "\n\n".join(context_blocks)
        # Call the OpenAI chat completions API.
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,  # A low temperature makes the output more deterministic and factual.
            messages=[
                # The system prompt sets the AI's persona and rules.
                {"role": "system", "content": SYSTEM_PROMPT},
                # The user prompt contains the context and the actual question.
                {"role": "user", "content": f"{prompt}\n\nQuestion: {question}"},
            ],
        )
        # Extract the text content from the first choice in the response.
        answer = response.choices[0].message.content or ""
        clean_answer = answer.strip()
        # As a safeguard, if the LLM returns an empty string, fall back to the standard "I don't know" response.
        if not clean_answer:
            LOGGER.warning("Received empty answer from LLM; falling back.")
            return "I don’t know from the documents."
        return clean_answer


# A helper function to format a single piece of retrieved context into a standardized block.
# This format makes it clear to the LLM where each piece of information comes from.
def build_context_block(doc_title: str, page_start: int, page_end: int, chunk_text: str) -> str:
    # Trim the chunk to a reasonable token limit to avoid exceeding the model's context window.
    trimmed_chunk = trim_to_token_limit(chunk_text, max_tokens=700)
    # Create a header that includes the document title and page range for citation.
    header = f"[Title: {doc_title}, Pages: {page_start}-{page_end}]"
    # Combine the header and the text, separated by a clear marker.
    return f"{header}\n{trimmed_chunk}\n---"
