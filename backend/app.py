"""
Flask API exposing the verification, scoring, and chat services.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
import threading
import uuid

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

from author_ai.pipelines.accuracy import AccuracyPipeline
from author_ai.pipelines.credibility import CredibilityPipeline
from author_ai.pipelines.validity import ValidityPipeline
from author_ai.pipelines.chat import ChatService
from author_ai.config import get_settings
from author_ai.services.environment import reset_environment
from author_ai.services.vector_store import VectorStore
from author_ai.services.file_store import save_upload
from author_ai.models import _now_iso
from author_ai.services.logger import setup_logger
from author_ai.services.recommendations import RecommendationService

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

logger = setup_logger(__name__)

# H5: Instantiate pipelines once at module level so they are reused across requests.
_accuracy_pipeline = AccuracyPipeline()
_credibility_pipeline = CredibilityPipeline()
_validity_pipeline = ValidityPipeline()
_chat_service = ChatService()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    settings = get_settings()
    # H5: Reuse module-level pipeline instances instead of creating new ones per request.
    accuracy = _accuracy_pipeline
    credibility = _credibility_pipeline
    validity = _validity_pipeline
    chat_service = _chat_service
    recommendation_service = RecommendationService()
    repo = accuracy.repo

    def serialize_upload(upload: dict) -> Dict[str, Any]:
        return {
            "upload_id": upload["upload_id"],
            "file_name": upload["file_name"],
            "file_type": upload["file_type"],
            "path": upload["path"],
            "created_at": upload.get("created_at"),
            "file_url": f"/api/uploads/{upload['upload_id']}/file",
        }

    def store_files(file_type: str, storages) -> list[Dict[str, Any]]:
        uploads = []
        for storage in storages:
            if not storage.filename:
                continue
            upload_id, path = save_upload(storage)
            record = {
                "upload_id": upload_id,
                "file_name": secure_filename(storage.filename) or upload_id,
                "file_type": file_type,
                "path": str(path),
                "created_at": _now_iso(),
            }
            repo.add_upload(record)
            uploads.append(serialize_upload(record))
        if not uploads:
            raise ValueError("No files were provided.")
        return uploads

    def build_sources_payload(source_uploads: list[dict], usage_map: Optional[Dict[str, int]] = None):
        usage_map = usage_map or {}
        payload = []
        for upload in source_uploads:
            credibility_record = repo.get_credibility(upload["upload_id"])
            cred_fraction = _score_fraction(credibility_record["score"]) if credibility_record else 0.0
            usage_count = usage_map.get(upload["upload_id"], 0)
            source_meta = serialize_upload(upload)
            payload.append(
                {
                    "id": upload["upload_id"],
                    "name": source_meta["file_name"],
                    "file_url": source_meta["file_url"],
                    "summary": f"{usage_count} supporting claims identified." if usage_count else "No direct claims yet.",
                    "scores": {"credibility": cred_fraction},
                    "usage_count": usage_count,
                }
            )
        return payload

    def normalize_recommendation(record: dict) -> dict:
        title = record.get("title") or record.get("name") or "Source"
        summary = record.get("summary") or record.get("abstract")
        summary = summary or "Summary unavailable."
        return {
            "id": record.get("id"),
            "title": title,
            "summary": summary,
            "abstract": record.get("abstract"),
            "credibility_score": record.get("credibility_score"),
            "validity_score": record.get("validity_score"),
            "date_published": record.get("date_published") or record.get("publication_year"),
            "authors": record.get("authors") or [],
            "doi": record.get("doi"),
            "url": record.get("url"),
            "openalex_url": record.get("openalex_url"),
            "host_venue": record.get("host_venue"),
        }

    def _score_fraction(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        return max(0.0, min(float(value) / 100.0, 1.0))

    def build_report_summary(job: dict) -> Dict[str, Any]:
        result = job.get("result_json") or {}
        claims = result.get("claims", [])
        total_claims = len(claims)
        supported_claims = sum(1 for claim in claims if claim.get("verdict") == "SUPPORTED")
        accuracy_score = supported_claims / total_claims if total_claims else 0.0

        validity_scores = result.get("validity", {})
        validity_score = _score_fraction(validity_scores.get("overall"))

        credibility_summary = result.get("credibility") or {}
        credibility_score = _score_fraction(credibility_summary.get("overall"))

        overall_components = [accuracy_score, credibility_score, validity_score]
        overall = sum(overall_components) / len(overall_components) if overall_components else 0.0

        contradicted_claims = sum(1 for claim in claims if claim.get("verdict") == "CONTRADICTED")
        not_found_claims = total_claims - supported_claims - contradicted_claims

        report_upload = repo.get_upload(job.get("report_id")) if job.get("report_id") else None
        report_info = (
            serialize_upload(report_upload)
            if report_upload
            else {"upload_id": job.get("report_id"), "file_name": "Report", "file_url": None}
        )

        source_uploads = [repo.get_upload(source_id) for source_id in job.get("source_ids", [])]
        source_uploads = [upload for upload in source_uploads if upload]

        usage_rows = repo.source_usage(job.get("report_id")) if job.get("report_id") else []
        usage_map = {row["source_id"]: row["usage_count"] for row in usage_rows}

        sources_payload = build_sources_payload(source_uploads, usage_map)
        recommended = sorted(
            sources_payload,
            key=lambda entry: (entry["usage_count"], entry["scores"].get("credibility", 0.0)),
            reverse=True,
        )
        top_sources = [
            {
                "id": entry["id"],
                "name": entry["name"],
                "usage_count": entry["usage_count"],
                "credibility": entry["scores"].get("credibility", 0),
            }
            for entry in recommended[:5]
        ]

        recommended_sources = result.get("recommended_sources")
        recommendations_persisted = True
        if not recommended_sources:
            recommendations_persisted = False
            recommended_sources = recommendation_service.recommend(
                claims=claims,
                existing_sources=sources_payload,
                report_title=report_info["file_name"],
                limit=5,
            )
        if not recommended_sources:
            recommended_sources = []
        else:
            recommended_sources = [normalize_recommendation(item) for item in recommended_sources]
        if not recommendations_persisted:
            updated_result = dict(result)
            updated_result["recommended_sources"] = recommended_sources
            repo.update_job(
                job["job_id"],
                result_json=updated_result,
                updated_at=_now_iso(),
            )

        chat_messages = []
        for idx, claim in enumerate(claims[:3]):
            chat_messages.append(
                {
                    "id": idx + 1,
                    "author": "System",
                    "text": f"{claim.get('verdict', 'UNKNOWN')}: {claim.get('text', '')[:220]}",
                }
            )
        if not chat_messages:
            chat_messages.append(
                {
                    "id": 1,
                    "author": "System",
                    "text": "Pipeline complete. Ask about specific claims or sources.",
                }
            )

        report_summary_text = (
            f"{supported_claims} of {total_claims} claims currently supported." if total_claims else "Pipeline results pending."
        )

        return {
            "job_id": job["job_id"],
            "report": {
                "id": report_info["upload_id"],
                "name": report_info["file_name"],
                "pdf_url": report_info["file_url"],
                "summary": report_summary_text,
            },
            "scores": {
                "overall": overall,
                "accuracy": accuracy_score,
                "credibility": credibility_score,
                "validity": validity_score,
            },
            "recommended_sources": recommended_sources,
            "chat_messages": chat_messages,
            "sources": sources_payload,
            "claims": claims,
            "stats": {
                "claims_total": total_claims,
                "claims_supported": supported_claims,
                "claims_contradicted": contradicted_claims,
                "claims_not_found": max(not_found_claims, 0),
            },
            "top_sources": top_sources,
        }

    def build_source_detail(source_id: str, *, limit: int = 5, page: int = 0) -> Dict[str, Any]:
        upload = repo.get_upload(source_id)
        if not upload:
            raise ValueError("Source not found")
        source_meta = serialize_upload(upload)
        doc = repo.get_document(source_id)
        metadata = doc.get("metadata") if doc else {}
        summary_text = metadata.get("summary") if metadata else ""
        if not summary_text and doc and doc.get("body_text"):
            summary_text = (doc["body_text"] or "").split("\n")[0][:400]
        summary_text = summary_text or source_meta["file_name"]
        credibility_record = repo.get_credibility(source_id)
        offset = max(page, 0) * max(limit, 1)
        claims = repo.get_claims_for_source(source_id, limit=limit, offset=offset)
        total_claims = repo.count_claims_for_source(source_id)
        has_more_claims = offset + len(claims) < total_claims
        supported_claims = sum(1 for claim in claims if claim.get("verdict") == "SUPPORTED")
        validity_info = (
            {
                "score": supported_claims / len(claims) if claims else 0.0,
                "supported": supported_claims,
                "total": len(claims),
            }
            if claims
            else None
        )
        tables_preview = (metadata.get("table_preview") or [])[:1]
        return {
            "upload": source_meta,
            "credibility": credibility_record,
            "claims": claims,
            "claim_total": total_claims,
            "claim_has_more": has_more_claims,
            "usage_count": total_claims,
            "tables": tables_preview,
            "summary": summary_text,
            "validity": validity_info,
        }

    def require_api_key(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            expected = settings.api_key
            provided = request.headers.get("X-API-Key")
            if expected and provided != expected:
                return jsonify({"error": "Unauthorized"}), 401
            return func(*args, **kwargs)

        return wrapper

    @app.errorhandler(FileNotFoundError)
    def handle_not_found(error):
        return jsonify({"error": str(error)}), 404

    @app.errorhandler(ValueError)
    def handle_value_error(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(Exception)
    def handle_exception(error):
        if isinstance(error, HTTPException):
            return jsonify({"error": error.description}), error.code
        return jsonify({"error": "Internal Server Error"}), 500

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return jsonify({"status": "ok"})

    @app.post("/api/uploads/source")
    @require_api_key
    def upload_source_files():
        storages = request.files.getlist("files") or request.files.getlist("file")
        uploads = store_files("SOURCE", storages)
        return jsonify({"uploads": uploads})

    @app.post("/api/uploads/report")
    @require_api_key
    def upload_report_file():
        storage = request.files.get("file")
        if not storage:
            raise ValueError("Report file is required")
        upload = store_files("REPORT", [storage])[0]
        return jsonify(upload)

    @app.get("/api/uploads")
    @require_api_key
    def list_uploads():
        file_type = request.args.get("type")
        records = repo.list_uploads(file_type.upper()) if file_type else repo.list_uploads()
        return jsonify({"uploads": [serialize_upload(record) for record in records]})

    @app.delete("/api/uploads/<upload_id>")
    @require_api_key
    def delete_upload(upload_id: str):
        upload = repo.get_upload(upload_id)
        if not upload:
            return jsonify({"status": "not_found"}), 404
        path = Path(upload["path"])
        if path.exists():
            path.unlink()
            parent = path.parent
            try:
                if parent.exists():
                    parent.rmdir()
            except OSError:
                pass
        repo.delete_upload(upload_id)
        return jsonify({"status": "deleted"})

    @app.get("/api/uploads/<upload_id>/file")
    @require_api_key
    def download_upload(upload_id: str):
        upload = repo.get_upload(upload_id)
        if not upload:
            return jsonify({"error": "File not found"}), 404
        return send_file(upload["path"], as_attachment=False)

    @app.get("/api/dashboard")
    @require_api_key
    def dashboard():
        # M15: replaced hardcoded mock data with real DB queries.
        job = repo.get_latest_job()
        if not job:
            return jsonify({"error": "No completed reports"}), 404
        return jsonify(build_report_summary(job))

    @app.post("/api/run_pipeline")
    @require_api_key
    def run_pipeline():
        payload = request.get_json(force=True)
        source_ids = payload.get("source_ids") or []
        report_id = payload.get("report_id")
        if not source_ids:
            raise ValueError("source_ids is required")
        if not report_id:
            raise ValueError("report_id is required")

        job_id = str(uuid.uuid4())
        repo.create_job(
            {
                "job_id": job_id,
                "status": "QUEUED",
                "report_id": report_id,
                "source_ids": source_ids,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        )

        source_uploads = []
        for source_id in source_ids:
            upload = repo.get_upload(source_id)
            if not upload or upload["file_type"].upper() != "SOURCE":
                raise ValueError(f"Invalid source upload: {source_id}")
            source_uploads.append(upload)

        report_upload = repo.get_upload(report_id)
        if not report_upload or report_upload["file_type"].upper() != "REPORT":
            raise ValueError("Invalid report upload")

        # H6: Run the pipeline in a background thread so this request returns immediately.
        def _run_pipeline_job():
            def _progress(step: str, label: str, status: str = "done"):
                repo.push_job_progress(job_id, step, label, status)

            try:
                repo.update_job(job_id, status="RUNNING", updated_at=_now_iso())
                reset_environment()
                accuracy.vector_store = VectorStore("sources")

                n = len(source_uploads)
                _progress("indexing", f"Indexing {n} source{'s' if n != 1 else ''}…", "running")
                for upload in source_uploads:
                    accuracy.index_source(upload)
                    credibility.score_source(upload)
                _progress("indexing", f"Indexed {n} source{'s' if n != 1 else ''}")

                _progress("verifying", "Extracting claims and retrieving evidence…", "running")
                verification = accuracy.verify_report(report_upload)
                _progress("verifying", f"Verified {len(verification.get('claims', []))} claims")

                _progress("validity", "Scoring validity…", "running")
                validity_scores = validity.score_report(Path(report_upload["path"]), report_upload["upload_id"])
                _progress("validity", "Validity scored")

                _progress("credibility", "Aggregating credibility scores…", "running")
                credibility_summary = credibility.aggregate_report(verification["report_id"])
                _progress("credibility", "Credibility aggregated")

                _progress("recommendations", "Finding recommended sources…", "running")
                usage_rows = repo.source_usage(verification["report_id"])
                usage_map = {row["source_id"]: row["usage_count"] for row in usage_rows}
                sources_payload = build_sources_payload(source_uploads, usage_map)
                recommended_sources = recommendation_service.recommend(
                    claims=verification["claims"],
                    existing_sources=sources_payload,
                    report_title=report_upload["file_name"],
                    limit=5,
                )
                if not recommended_sources:
                    recommended_sources = []
                else:
                    recommended_sources = [normalize_recommendation(item) for item in recommended_sources]
                _progress("recommendations", f"Found {len(recommended_sources)} recommendation{'s' if len(recommended_sources) != 1 else ''}")

                result_payload = {
                    "claims": verification["claims"],
                    "report_id": verification["report_id"],
                    "validity": validity_scores.__dict__,
                    "credibility": credibility_summary,
                    "recommended_sources": recommended_sources,
                }

                repo.update_job(
                    job_id,
                    status="DONE",
                    result_json=result_payload,
                    updated_at=_now_iso(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pipeline job %s failed", job_id)
                # Mark the last running step as failed
                job = repo.get_job(job_id)
                if job:
                    for entry in reversed(job.get("progress_json") or []):
                        if entry.get("status") == "running":
                            repo.push_job_progress(job_id, entry["step"], entry["label"], "failed")
                            break
                repo.update_job(
                    job_id,
                    status="FAILED",
                    error_message=str(exc),
                    updated_at=_now_iso(),
                )

        thread = threading.Thread(target=_run_pipeline_job, daemon=True)
        thread.start()

        return jsonify({"job_id": job_id, "status": "QUEUED"})

    @app.get("/api/jobs/<job_id>")
    @require_api_key
    def get_job(job_id: str):
        job = repo.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)

    @app.get("/api/reports/latest")
    @require_api_key
    def get_latest_report_summary():
        job = repo.get_latest_job()
        if not job:
            return jsonify({"error": "No completed reports"}), 404
        return jsonify(build_report_summary(job))

    @app.get("/api/reports/<job_id>/summary")
    @require_api_key
    def get_report_summary(job_id: str):
        job = repo.get_job(job_id)
        if not job or job.get("status") != "DONE":
            return jsonify({"error": "Report not ready"}), 404
        return jsonify(build_report_summary(job))

    @app.get("/api/reports/<job_id>/claims")
    @require_api_key
    def get_report_claims(job_id: str):
        job = repo.get_job(job_id)
        if not job or job.get("status") != "DONE":
            return jsonify({"error": "Report not ready"}), 404
        report_id = job.get("report_id")
        if not report_id:
            return jsonify({"claims": [], "total": 0, "has_more": False})
        try:
            limit = int(request.args.get("limit", 5))
            page = int(request.args.get("page", 0))
        except ValueError:
            return jsonify({"error": "Invalid pagination parameters"}), 400
        offset = max(page, 0) * max(limit, 1)
        claims = repo.list_claims_by_report(report_id, limit=limit, offset=offset)
        total = repo.count_claims_by_report(report_id)
        has_more = offset + len(claims) < total
        # Build a lookup of upload_id -> upload record for source name resolution
        source_ids = job.get("source_ids") or []
        uploads_by_id = {}
        for sid in source_ids:
            upload = repo.get_upload(sid)
            if upload:
                uploads_by_id[sid] = upload
        # Enrich each claim with its top evidence snippets and report page
        for claim in claims:
            import json as _json
            raw_meta = claim.get("metadata") or {}
            if isinstance(raw_meta, str):
                try:
                    raw_meta = _json.loads(raw_meta)
                except Exception:
                    raw_meta = {}
            claim["parent_page"] = raw_meta.get("parent_page")
            evidence_rows = repo.list_evidence_for_claim(claim["claim_id"])
            primary_evidence = [e for e in evidence_rows if e.get("verdict_label") != "ALTERNATIVE"]
            claim["evidence"] = [
                {
                    "snippet": (e.get("metadata") or {}).get("snippet", ""),
                    "source_id": e.get("source_id"),
                    "source_name": uploads_by_id.get(e.get("source_id"), {}).get("file_name", e.get("source_id")),
                    "page": ((e.get("metadata") or {}).get("parent") or {}).get("page"),
                    "score": (e.get("metadata") or {}).get("score"),
                }
                for e in primary_evidence[:2]  # max 2 evidence snippets per claim
            ]
        return jsonify({"claims": claims, "total": total, "has_more": has_more})

    @app.get("/api/sources/<source_id>")
    @require_api_key
    def get_source_detail(source_id: str):
        upload = repo.get_upload(source_id)
        if not upload:
            return jsonify({"error": "Source not found"}), 404
        try:
            limit = int(request.args.get("claim_limit", 5))
            page = int(request.args.get("claim_page", 0))
        except ValueError:
            return jsonify({"error": "Invalid pagination parameters"}), 400
        detail = build_source_detail(source_id, limit=limit, page=page)
        return jsonify(detail)

    @app.get("/api/claims")
    @require_api_key
    def list_claims():
        claims = accuracy.repo.list_claims()
        return jsonify({"claims": claims})

    @app.post("/api/chat")
    @require_api_key
    def chat():
        payload = request.get_json(force=True)
        question = payload.get("question", "")
        session_id = payload.get("session_id")
        job_id = payload.get("job_id")
        mode = payload.get("mode")
        mode_locked = bool(payload.get("mode_locked", False))
        report_id = None
        if job_id:
            job = repo.get_job(job_id)
            if not job or job.get("status") != "DONE":
                return jsonify({"error": "Report not ready"}), 400
            report_id = job.get("report_id")
        else:
            latest = repo.get_latest_job()
            if latest:
                report_id = latest.get("report_id")
        if not report_id:
            return jsonify({"error": "No completed report"}), 400

        response = chat_service.respond(
            question,
            report_id=report_id,
            session_id=session_id,
            mode=mode,
            mode_locked=mode_locked,
        )
        return jsonify(response)

    @app.get("/api/chat/history")
    @require_api_key
    def chat_history():
        job_id = request.args.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id required"}), 400
        job = repo.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        report_id = job.get("report_id")
        if not report_id:
            return jsonify({"history": []})
        history = repo.get_chat_history(report_id)
        return jsonify({"history": history})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5001)
