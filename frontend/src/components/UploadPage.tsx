import { ChangeEvent, FormEvent, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useReportData } from "../context/ReportDataContext";
import { runPipelineWithUploads, fetchJob, ProgressEntry } from "../api/client";

const PIPELINE_STEPS: { step: string; label: string }[] = [
  { step: "indexing",        label: "Indexing sources" },
  { step: "verifying",       label: "Verifying report claims" },
  { step: "validity",        label: "Scoring validity" },
  { step: "credibility",     label: "Aggregating credibility" },
  { step: "recommendations", label: "Finding recommendations" },
];

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

function stepIcon(status: ProgressEntry["status"] | "pending") {
  if (status === "done")    return "✓";
  if (status === "running") return "⟳";
  if (status === "failed")  return "✗";
  return "·";
}

function PipelineProgress({ entries }: { entries: ProgressEntry[] }) {
  const byStep = Object.fromEntries(entries.map((e) => [e.step, e]));
  return (
    <ul className="pipeline-progress">
      {PIPELINE_STEPS.map(({ step, label }) => {
        const entry = byStep[step];
        const status = entry?.status ?? "pending";
        const displayLabel = entry?.label ?? label;
        return (
          <li
            key={step}
            className={`pipeline-progress__step pipeline-progress__step--${status}`}
          >
            <span className="pipeline-progress__icon">{stepIcon(status)}</span>
            <span>{displayLabel}</span>
          </li>
        );
      })}
    </ul>
  );
}

// UploadPage lets the user add sources and the primary report before entering the dashboard.
const UploadPage = () => {
  const {
    internalSources,
    addInternalSource,
    removeInternalSource,
    reportDocument,
    setReportDocument,
    refreshUploads,
    setSummaryData,
    setJobStatus,
    refreshJobStatus
  } = useReportData();
  const navigate = useNavigate();

  const sourceInputRef = useRef<HTMLInputElement>(null);
  const reportInputRef = useRef<HTMLInputElement>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pipelineMessage, setPipelineMessage] = useState<string | null>(null);
  const [progressEntries, setProgressEntries] = useState<ProgressEntry[]>([]);

  const handleSourceAdd = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        await addInternalSource(file);
      } catch (error) {
        setSubmitError(error instanceof Error ? error.message : "Failed to upload source.");
      }
    }
    event.target.value = "";
  };

  const handleReportAdd = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        await setReportDocument(file);
      } catch (error) {
        setSubmitError(error instanceof Error ? error.message : "Failed to upload report.");
      }
    }
    event.target.value = "";
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitError(null);

    if (!internalSources.length) {
      setSubmitError("Please add at least one source PDF to run the pipeline.");
      return;
    }

    if (!reportDocument?.id) {
      setSubmitError("Please upload the report PDF before continuing.");
      return;
    }

    setIsSubmitting(true);
    try {
      const job = await runPipelineWithUploads(
        internalSources.map((source) => source.id),
        reportDocument.id
      );
      setPipelineMessage(`Pipeline status: ${job.status}`);
      setJobStatus({ job_id: job.job_id, status: job.status });
      await monitorJob(job.job_id);
    } catch (error) {
      setSubmitError(
        error instanceof Error ? error.message : "Failed to run the pipeline. Please try again."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const monitorJob = async (jobId: string) => {
    const POLL_INTERVAL = 2000;
    let done = false;
    while (!done) {
      const job = await fetchJob(jobId);
      setJobStatus({ job_id: job.job_id, status: job.status, updated_at: job.updated_at });
      setPipelineMessage(`Pipeline status: ${job.status}`);
      if (job.progress_json?.length) {
        setProgressEntries(job.progress_json);
      }
      if (job.status === "DONE") {
        localStorage.setItem("active_job_id", job.job_id);
        localStorage.setItem("last_run_at", job.updated_at ?? new Date().toISOString());
        setSummaryData(null);
        await refreshUploads();
        await refreshJobStatus(job.job_id);
        done = true;
        navigate("/dashboard");
      } else if (job.status === "FAILED") {
        setSubmitError(`Pipeline failed: ${job.error_message ?? "check server logs"}`);
        done = true;
      } else {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
      }
    }
  };

  return (
    <div className="upload">
      <form className="upload__form" onSubmit={handleSubmit}>
        <div className="upload__header">
          <h1 className="upload__title">Upload Sources and Report</h1>
          <p className="upload__subtitle">Add your source documents and the report to verify.</p>
        </div>
        <div className="upload__columns">
          {/* Sources panel */}
          <section className="upload__panel card">
            <header className="card__header">
              <h2>Sources</h2>
              {internalSources.length > 0 && (
                <span className="upload__count">{internalSources.length}</span>
              )}
            </header>
            {internalSources.length > 0 && (
              <div className="upload__list">
                {internalSources.map((source) => (
                  <div key={source.id} className="upload__item upload__item--with-actions">
                    <span className="upload__item-icon"><PdfIcon size={15} /></span>
                    <span className="upload__item-name">{source.name}</span>
                    <button
                      type="button"
                      className="upload__remove"
                      aria-label={`Remove ${source.name}`}
                      onClick={async () => {
                        try {
                          await removeInternalSource(source.id);
                        } catch (error) {
                          setSubmitError(
                            error instanceof Error ? error.message : "Failed to remove source."
                          );
                        }
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
            <button
              type="button"
              className={`upload__dropzone${internalSources.length === 0 ? " upload__dropzone--fill" : ""}`}
              onClick={() => sourceInputRef.current?.click()}
            >
              <span className="upload__dropzone-icon"><UploadIcon /></span>
              <span className="upload__dropzone-text">Click to add a PDF source</span>
              <span className="upload__dropzone-hint">PDF files only</span>
            </button>
            <input
              ref={sourceInputRef}
              type="file"
              accept=".pdf"
              hidden
              onChange={handleSourceAdd}
            />
          </section>

          {/* Report panel */}
          <section className="upload__panel card">
            <header className="card__header">
              <h2>Report</h2>
            </header>
            <div className="upload__list upload__list--single">
              {reportDocument?.name ? (
                <div className="upload__item upload__item--report">
                  <span className="upload__item-icon upload__item-icon--accent"><PdfIcon size={15} /></span>
                  <span className="upload__item-name">{reportDocument.name}</span>
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
              {reportDocument?.name && (
                <button
                  type="button"
                  className="upload__replace"
                  onClick={() => reportInputRef.current?.click()}
                >
                  Replace Report
                </button>
              )}
              <input
                ref={reportInputRef}
                type="file"
                accept=".pdf"
                hidden
                onChange={handleReportAdd}
              />
            </div>

            {submitError ? <p className="upload__status upload__status--error">{submitError}</p> : null}
            {isSubmitting && progressEntries.length === 0 ? (
              <p className="upload__status">Starting pipeline…</p>
            ) : null}
            {isSubmitting && progressEntries.length > 0 ? (
              <PipelineProgress entries={progressEntries} />
            ) : null}

            <button type="submit" className="upload__submit" disabled={isSubmitting}>
              ➤
            </button>
          </section>
        </div>
      </form>
    </div>
  );
};

export default UploadPage;
