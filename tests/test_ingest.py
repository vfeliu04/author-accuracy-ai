from __future__ import annotations

import io


def test_ingest_reports_counts(client_with_stubs, dummy_pdf_bytes):
    client, _store = client_with_stubs
    data = {"files": (io.BytesIO(dummy_pdf_bytes), "dummy.pdf")}
    response = client.post("/ingest", data=data, content_type="multipart/form-data")
    body = response.get_json()

    assert response.status_code == 200
    assert body["ok"] is True
    assert body["chunks_added"] >= 1
    assert body["chunks_skipped"] == 0

    response_second = client.post("/ingest", data=data, content_type="multipart/form-data")
    body_second = response_second.get_json()
    assert response_second.status_code == 200
    assert body_second["chunks_added"] == 0
    assert body_second["chunks_skipped"] >= 1

