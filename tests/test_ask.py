from __future__ import annotations

import io

from mvp_rag.app import FALLBACK_ANSWER


def test_ask_returns_answer_and_citations(client_with_stubs, dummy_pdf_bytes):
    client, _store = client_with_stubs
    data = {"files": (io.BytesIO(dummy_pdf_bytes), "dummy.pdf")}
    client.post("/ingest", data=data, content_type="multipart/form-data")

    response = client.post("/ask", json={"question": "What greeting is included?"})
    body = response.get_json()

    assert response.status_code == 200
    assert body["ok"] is True
    assert body["answer"] != FALLBACK_ANSWER
    assert 0 < len(body["citations"]) <= 4


def test_ask_returns_fallback_without_context(client_with_stubs):
    client, store = client_with_stubs
    store.clear()
    response = client.post("/ask", json={"question": "Is there any data?"})
    body = response.get_json()

    assert response.status_code == 200
    assert body["ok"] is True
    assert body["answer"] == FALLBACK_ANSWER
    assert body["citations"] == []

