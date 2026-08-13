import { ChangeEvent, FormEvent, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useCreateRun } from "../api/queries";

function PdfIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="16" y1="13" x2="8" y2="13"/>
      <line x1="16" y1="17" x2="8" y2="17"/>
      <polyline points="10 9 9 9 8 9"/>
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width={28} height={28} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 16 12 12 8 16"/>
      <line x1="12" y1="12" x2="12" y2="21"/>
      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
    </svg>
  );
}

// A run is created in ONE request: the report + its sources are uploaded
// together (POST /api/runs), then we navigate to the run's page to watch the
// pipeline. Files are held in local state until submit — v2 has no separate
// per-file upload endpoint.
const UploadPage = () => {
  const navigate = useNavigate();
  const createRun = useCreateRun();

  const sourceInputRef = useRef<HTMLInputElement>(null);
  const reportInputRef = useRef<HTMLInputElement>(null);
  const [sources, setSources] = useState<File[]>([]);
  const [report, setReport] = useState<File | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const handleSourceAdd = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSources((prev) => [...prev, file]);
    }
    event.target.value = "";
  };

  const handleReportAdd = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setReport(file);
    }
    event.target.value = "";
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);
    if (!sources.length) {
      setFormError("Add at least one source PDF.");
      return;
    }
    if (!report) {
      setFormError("Upload the report PDF before continuing.");
      return;
    }
    createRun.mutate(
      { report, sources },
      { onSuccess: (data) => navigate(`/runs/${data.run_id}`) }
    );
  };

  const submitting = createRun.isPending;
  const error = formError ?? (createRun.error instanceof Error ? createRun.error.message : null);

  return (
    <div className="upload">
      <form className="upload__form" onSubmit={handleSubmit}>
        <div className="upload__header">
          <h1 className="upload__title">Upload Sources and Report</h1>
          <p className="upload__subtitle">Add your source documents and the report to verify.</p>
          <button
            type="button"
            className="upload__replace"
            onClick={() => navigate("/runs")}
          >
            View run history →
          </button>
        </div>
        <div className="upload__columns">
          <section className="upload__panel card">
            <header className="card__header">
              <h2>Sources</h2>
              {sources.length > 0 && <span className="upload__count">{sources.length}</span>}
            </header>
            {sources.length > 0 && (
              <div className="upload__list">
                {sources.map((source, index) => (
                  <div key={`${source.name}-${index}`} className="upload__item upload__item--with-actions">
                    <span className="upload__item-icon"><PdfIcon size={15} /></span>
                    <span className="upload__item-name">{source.name}</span>
                    <button
                      type="button"
                      className="upload__remove"
                      aria-label={`Remove ${source.name}`}
                      onClick={() => setSources((prev) => prev.filter((_, i) => i !== index))}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
            <button
              type="button"
              className={`upload__dropzone${sources.length === 0 ? " upload__dropzone--fill" : ""}`}
              onClick={() => sourceInputRef.current?.click()}
            >
              <span className="upload__dropzone-icon"><UploadIcon /></span>
              <span className="upload__dropzone-text">Click to add a PDF source</span>
              <span className="upload__dropzone-hint">PDF files only</span>
            </button>
            <input ref={sourceInputRef} type="file" accept=".pdf" hidden onChange={handleSourceAdd} />
          </section>

          <section className="upload__panel card">
            <header className="card__header">
              <h2>Report</h2>
            </header>
            <div className="upload__list upload__list--single">
              {report ? (
                <div className="upload__item upload__item--report">
                  <span className="upload__item-icon upload__item-icon--accent"><PdfIcon size={15} /></span>
                  <span className="upload__item-name">{report.name}</span>
                </div>
              ) : (
                <button
                  type="button"
                  className="upload__dropzone upload__dropzone--report"
                  onClick={() => reportInputRef.current?.click()}
                >
                  <span className="upload__dropzone-icon"><UploadIcon /></span>
                  <span className="upload__dropzone-text">Click to upload report</span>
                  <span className="upload__dropzone-hint">PDF files only</span>
                </button>
              )}
              {report && (
                <button
                  type="button"
                  className="upload__replace"
                  onClick={() => reportInputRef.current?.click()}
                >
                  Replace Report
                </button>
              )}
              <input ref={reportInputRef} type="file" accept=".pdf" hidden onChange={handleReportAdd} />
            </div>

            {error ? <p className="upload__status upload__status--error">{error}</p> : null}
            {submitting ? <p className="upload__status">Creating run…</p> : null}

            <button type="submit" className="upload__submit" disabled={submitting}>
              ➤
            </button>
          </section>
        </div>
      </form>
    </div>
  );
};

export default UploadPage;
