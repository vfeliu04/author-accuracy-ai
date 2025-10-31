"""Flask entrypoint for the hallucination checker web UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from flask import Flask, flash, redirect, render_template, url_for
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from flask_wtf.file import FileAllowed, FileField, MultipleFileField
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from wtforms import SelectField, SelectMultipleField, SubmitField

from src.hallcheck.config import settings
from src.hallcheck.webbridge import (
    clear_explanation,
    fetch_claim_detail,
    fetch_results,
    get_or_make_explanation,
    get_report_details,
    list_report_pdfs,
    list_source_pdfs,
    run_index,
    run_verify,
)


ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")
CSRFProtect(app)

uploads_dir = (settings.project_root or Path(__file__).resolve().parent) / "data" / "inputs"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(uploads_dir)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload cap
sources_uploads_dir = uploads_dir / "sources"
reports_uploads_dir = uploads_dir / "reports"
sources_uploads_dir.mkdir(parents=True, exist_ok=True)
reports_uploads_dir.mkdir(parents=True, exist_ok=True)

STATUS_BADGES = {
    "SUPPORTED": "bg-success",
    "CONTRADICTED": "bg-danger",
    "NOT_FOUND": "bg-secondary",
}


class IndexSourcesForm(FlaskForm):
    uploads = MultipleFileField(
        "Upload source PDFs",
        validators=[FileAllowed(ALLOWED_EXTENSIONS, "PDF files only.")],
        render_kw={"accept": ".pdf", "multiple": True},
    )
    existing = SelectMultipleField(
        "Or select existing source PDFs",
        choices=[],
        coerce=str,
        render_kw={"size": 6},
    )
    submit = SubmitField("Index sources")


class VerifyReportForm(FlaskForm):
    upload = FileField(
        "Upload report PDF",
        validators=[FileAllowed(ALLOWED_EXTENSIONS, "PDF files only.")],
        render_kw={"accept": ".pdf"},
    )
    existing = SelectField(
        "Or select existing report PDF",
        choices=[],
        coerce=str,
        default="",
        validate_choice=False,
    )
    submit = SubmitField("Verify report")


class RegenerateExplanationForm(FlaskForm):
    submit = SubmitField("Regenerate explanation")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def save_uploaded_files(files: Iterable[FileStorage], target_dir: Optional[Path] = None) -> List[str]:
    saved_paths: List[str] = []
    destination_root = target_dir or Path(app.config["UPLOAD_FOLDER"])
    for storage in files:
        if not storage or not storage.filename:
            continue
        filename = secure_filename(storage.filename)
        if not filename or not allowed_file(filename):
            raise ValueError(f"Unsupported file type for '{storage.filename}'.")
        destination = destination_root / filename
        counter = 1
        stem = destination.stem
        suffix = destination.suffix
        while destination.exists():
            destination = destination_root / f"{stem}_{counter}{suffix}"
            counter += 1
        storage.save(destination)
        saved_paths.append(str(destination.resolve()))
    return saved_paths


def _prepare_index_form(form: IndexSourcesForm) -> None:
    choices = [(item["path"], item["label"]) for item in list_source_pdfs()]
    form.existing.choices = choices


def _prepare_verify_form(form: VerifyReportForm) -> None:
    choices = [("", "-- Select existing report --")]
    choices.extend((item["path"], item["label"]) for item in list_report_pdfs())
    form.existing.choices = choices


@app.context_processor
def inject_globals():
    return {"STATUS_BADGES": STATUS_BADGES}


@app.route("/", methods=["GET"])
def home():
    index_form = IndexSourcesForm()
    verify_form = VerifyReportForm()
    _prepare_index_form(index_form)
    _prepare_verify_form(verify_form)
    return render_template(
        "home.html",
        index_form=index_form,
        verify_form=verify_form,
    )


@app.route("/index", methods=["POST"])
def index_sources_route():
    form = IndexSourcesForm()
    _prepare_index_form(form)
    if not form.validate_on_submit():
        flash("Invalid submission. Please try again.", "danger")
        return redirect(url_for("home"))

    upload_paths: List[str] = []
    try:
        upload_paths = save_uploaded_files(form.uploads.data or [], target_dir=sources_uploads_dir)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("home"))

    selected_paths = form.existing.data or []
    all_paths = _dedupe(upload_paths + selected_paths)

    try:
        summary = run_index(all_paths, index_name="sources")
    except ValueError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("home"))
    except Exception as exc:  # pragma: no cover - defensive web handling
        flash(f"Indexing failed: {exc}", "danger")
        return redirect(url_for("home"))

    flash(
        f"Indexed {summary.get('number_of_docs', 0)} document(s) into '{summary.get('index_name', 'sources')}' "
        f"covering {summary.get('number_of_chunks', 0)} chunk(s).",
        "success",
    )
    return redirect(url_for("home"))


@app.route("/verify", methods=["POST"])
def verify_report_route():
    form = VerifyReportForm()
    _prepare_verify_form(form)
    if not form.validate_on_submit():
        flash("Invalid submission. Please try again.", "danger")
        return redirect(url_for("home"))

    report_path: Optional[str] = None
    try:
        uploaded_paths = save_uploaded_files([form.upload.data] if form.upload.data else [], target_dir=reports_uploads_dir)
        if uploaded_paths:
            report_path = uploaded_paths[0]
        elif form.existing.data:
            report_path = form.existing.data
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("home"))

    if not report_path:
        flash("Provide a report PDF by uploading or selecting an existing file.", "warning")
        return redirect(url_for("home"))

    try:
        report_doc_id = run_verify(report_path, index_name="sources", topk=5)
    except Exception as exc:  # pragma: no cover - defensive web handling
        flash(f"Verification failed: {exc}", "danger")
        return redirect(url_for("home"))

    return redirect(url_for("results", report_doc_id=report_doc_id))


@app.route("/results/<int:report_doc_id>", methods=["GET"])
def results(report_doc_id: int):
    details = get_report_details(report_doc_id)
    if not details:
        flash("Report not found. Please run verification again.", "warning")
        return redirect(url_for("home"))

    claims = fetch_results(report_doc_id)
    status_counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "NOT_FOUND": 0}
    for claim in claims:
        status = claim.get("verdict_status")
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts.setdefault("UNKNOWN", 0)
            status_counts["UNKNOWN"] += 1

    return render_template(
        "results.html",
        report=details,
        claims=claims,
        total_claims=len(claims),
        status_counts=status_counts,
    )


@app.get("/claim/<int:claim_id>")
def claim_detail(claim_id: int):
    try:
        detail = fetch_claim_detail(claim_id)
    except ValueError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("home"))
    except Exception as exc:  # pragma: no cover - defensive handling
        flash(f"Unable to load claim: {exc}", "danger")
        return redirect(url_for("home"))

    if not detail.get("explanation"):
        try:
            detail["explanation"] = get_or_make_explanation(claim_id)
        except Exception as exc:  # pragma: no cover - defensive handling
            flash(f"Explanation could not be generated: {exc}", "warning")

    report_meta = None
    report_doc_id = detail.get("report_doc_id")
    if report_doc_id is not None:
        report_meta = get_report_details(report_doc_id)

    regen_form = RegenerateExplanationForm()

    return render_template("claim_detail.html", detail=detail, report=report_meta, regen_form=regen_form)


@app.post("/claim/<int:claim_id>/regenerate")
def claim_regen(claim_id: int):
    form = RegenerateExplanationForm()
    if not form.validate_on_submit():
        flash("Invalid submission. Please try again.", "danger")
        return redirect(url_for("claim_detail", claim_id=claim_id))

    try:
        clear_explanation(claim_id)
    except ValueError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("claim_detail", claim_id=claim_id))
    except Exception as exc:  # pragma: no cover - defensive handling
        flash(f"Unable to reset explanation: {exc}", "danger")
        return redirect(url_for("claim_detail", claim_id=claim_id))

    try:
        get_or_make_explanation(claim_id)
    except Exception as exc:  # pragma: no cover - defensive handling
        flash(f"Explanation could not be generated: {exc}", "warning")
    else:
        flash("Explanation regenerated.", "success")

    return redirect(url_for("claim_detail", claim_id=claim_id))


if __name__ == "__main__":  # pragma: no cover
    app.run(host="127.0.0.1", port=5000, debug=True)
