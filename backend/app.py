"""
Flask API exposing the verification, scoring, and chat services.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
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

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

logger = setup_logger(__name__)

def _resolve_pdf_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return path


MOCK_DASHBOARD = {
    "report_title": "World - Hunger Report",
    "summary": (
        "The 2025 World Hunger and Food Chain Disruptions report highlights how climate shocks, "
        "conflict-driven displacement, and fragile logistics networks are converging to keep 735 million "
        "people in chronic food insecurity. It contrasts regions with resilient storage and cold-chain "
        "investments against those relying on volatile grain imports, underscoring the need for rapid "
        "response funds and nutrition-focused safety nets."
    ),
    "scores": {
        "overall": 0.78,
        "accuracy": 0.74,
        "credibility": 0.81,
        "validity": 0.69,
    },
    "recommended_sources": [
        "Global Food Resilience Index 2025",
        "Nutrition Equity Observatory Brief",
        "AgriSupply Chain Stability Outlook",
        "Climate Resilient Harvests 2024",
        "Urban Food Access Benchmark 2025",
        "FAO Logistics Pulse - June 2025",
    ],
    "chat_suggestions": [
        {"id": 1, "author": "System", "text": "Welcome back! Ask anything about improving this report."},
        {"id": 2, "author": "User", "text": "What sections should I revise first?"},
    ],
}


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    settings = get_settings()
    accuracy = AccuracyPipeline()
    credibility = CredibilityPipeline()
    validity = ValidityPipeline()
    chat_service = ChatService()
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

        sources_payload = []
        for upload in source_uploads:
            credibility_record = repo.get_credibility(upload["upload_id"])
            cred_fraction = _score_fraction(credibility_record["score"]) if credibility_record else 0.0
            usage_count = usage_map.get(upload["upload_id"], 0)
            source_meta = serialize_upload(upload)
            sources_payload.append(
                {
                    "id": upload["upload_id"],
                    "name": source_meta["file_name"],
                    "file_url": source_meta["file_url"],
                    "summary": f"{usage_count} supporting claims identified." if usage_count else "No direct claims yet.",
                    "scores": {"credibility": cred_fraction},
                    "usage_count": usage_count,
                }
            )

        recommended = sorted(
            sources_payload,
            key=lambda entry: (entry["usage_count"], entry["scores"].get("credibility", 0.0)),
            reverse=True,
        )
        recommended_sources = [entry["name"] for entry in recommended[:5]]
        top_sources = [
            {
                "id": entry["id"],
                "name": entry["name"],
                "usage_count": entry["usage_count"],
                "credibility": entry["scores"].get("credibility", 0),
            }
            for entry in recommended[:5]
        ]

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

    def build_source_detail(source_id: str) -> Dict[str, Any]:
        upload = repo.get_upload(source_id)
        if not upload:
            raise ValueError("Source not found")
        doc = repo.get_document(source_id)
        metadata = doc.get("metadata") if doc else {}
        summary_text = metadata.get("summary") if metadata else ""
        if not summary_text and doc and doc.get("body_text"):
            summary_text = (doc["body_text"] or "").split("\n")[0][:400]
        summary_text = summary_text or source_meta["file_name"]
        credibility_record = repo.get_credibility(source_id)
        claims = repo.get_claims_for_source(source_id)
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
        source_meta = serialize_upload(upload)
        return {
            "upload": source_meta,
            "credibility": credibility_record,
            "claims": claims,
            "usage_count": len(claims),
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

    @app.post("/api/ingest/source")
    @require_api_key
    def ingest_source():
        payload = request.get_json(force=True)
        pdf_path = _resolve_pdf_path(payload["path"])
        result = accuracy.index_source(pdf_path)
        return jsonify({"chunks": len(result["chunks"]), "tables": len(result["document"]["tables"])})

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
    def download_upload(upload_id: str):
        upload = repo.get_upload(upload_id)
        if not upload:
            return jsonify({"error": "File not found"}), 404
        return send_file(upload["path"], as_attachment=False)

    @app.post("/api/verify/report")
    @require_api_key
    def verify_report():
        payload = request.get_json(force=True)
        pdf_path = _resolve_pdf_path(payload["path"])
        verification = accuracy.verify_report(pdf_path)
        validity_scores = validity.score_report(pdf_path)
        credibility_summary = credibility.aggregate_report(verification["report_id"])
        return jsonify(
            {
                "claims": verification["claims"],
                "report_id": verification["report_id"],
                "validity": validity_scores.__dict__,
                "credibility": credibility_summary,
            }
        )

    @app.get("/api/dashboard")
    @require_api_key
    def dashboard():
        return jsonify(MOCK_DASHBOARD)

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

        try:
            repo.update_job(job_id, status="RUNNING", updated_at=_now_iso())
            reset_environment()
            accuracy.vector_store = VectorStore("sources")

            for upload in source_uploads:
                accuracy.index_source(upload)
                credibility.score_source(upload)

            verification = accuracy.verify_report(report_upload)
            validity_scores = validity.score_report(Path(report_upload["path"]), report_upload["upload_id"])
            credibility_summary = credibility.aggregate_report(verification["report_id"])

            result_payload = {
                "claims": verification["claims"],
                "report_id": verification["report_id"],
                "validity": validity_scores.__dict__,
                "credibility": credibility_summary,
            }

            repo.update_job(
                job_id,
                status="DONE",
                result_json=result_payload,
                updated_at=_now_iso(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline job %s failed", job_id)
            repo.update_job(
                job_id,
                status="FAILED",
                error_message=str(exc),
                updated_at=_now_iso(),
            )
            return (
                jsonify(
                    {
                        "job_id": job_id,
                        "status": "FAILED",
                        "error": "Pipeline execution failed.",
                        "details": str(exc),
                    }
                ),
                500,
            )

        return jsonify({"job_id": job_id, "status": "DONE", "result": result_payload})

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
            return jsonify({"claims": []})
        claims = repo.list_claims_by_report(report_id)
        return jsonify({"claims": claims})

    @app.get("/api/sources/<source_id>")
    @require_api_key
    def get_source_detail(source_id: str):
        upload = repo.get_upload(source_id)
        if not upload:
            return jsonify({"error": "Source not found"}), 404
        detail = build_source_detail(source_id)
        return jsonify(detail)

    @app.post("/api/credibility")
    @require_api_key
    def score_source():
        payload = request.get_json(force=True)
        pdf_path = _resolve_pdf_path(payload["path"])
        score = credibility.score_source(pdf_path)
        return jsonify(score.__dict__)

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

        response = chat_service.respond(question, report_id=report_id, session_id=session_id)
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
