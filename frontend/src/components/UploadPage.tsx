import { ChangeEvent, FormEvent, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useReportData } from "../context/ReportDataContext";
import { runPipelineWithUploads, fetchJob } from "../api/client";

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
      if (job.status === "DONE") {
        localStorage.setItem("active_job_id", job.job_id);
        localStorage.setItem("last_run_at", job.updated_at ?? new Date().toISOString());
        setSummaryData(null);
        await refreshUploads();
        await refreshJobStatus(job.job_id);
        done = true;
        navigate("/dashboard");
      } else if (job.status === "FAILED") {
        setSubmitError("Pipeline failed. Please check server logs.");
        done = true;
      } else {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
      }
    }
  };

  return (
    <div className="upload">
      <form className="upload__form" onSubmit={handleSubmit}>
        <h1 className="upload__title">Upload Sources and Report</h1>
        <div className="upload__columns">
          <section className="upload__panel card">
            <header className="card__header">
              <h2>Sources</h2>
            </header>
            <div className="upload__list">
              {internalSources.map((source) => (
                <div key={source.id} className="upload__item upload__item--with-actions">
                  <span>{source.name}</span>
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
              {internalSources.length === 0 ? (
                <span className="upload__placeholder">No sources added yet.</span>
              ) : null}
            </div>
            <button type="button" className="upload__button" onClick={() => sourceInputRef.current?.click()}>
              + Add Source
            </button>
            <input
              ref={sourceInputRef}
              type="file"
              accept=".pdf"
              hidden
              onChange={handleSourceAdd}
            />
          </section>
          <section className="upload__panel card">
            <header className="card__header">
              <h2>Report</h2>
            </header>
            <div className="upload__list upload__list--single">
              {reportDocument?.name ? (
                <span className="upload__item">{reportDocument.name}</span>
              ) : (
                <span className="upload__placeholder">No report uploaded.</span>
              )}
              <button
                type="button"
                className="upload__button"
                onClick={() => reportInputRef.current?.click()}
              >
                {reportDocument?.name ? "Replace Report" : "Add Report"}
              </button>
              <input
                ref={reportInputRef}
                type="file"
                accept=".pdf"
                hidden
                onChange={handleReportAdd}
              />
            </div>
            {submitError ? <p className="upload__status upload__status--error">{submitError}</p> : null}
            {pipelineMessage ? <p className="upload__status">{pipelineMessage}</p> : null}
            {isSubmitting ? <p className="upload__status">Running pipeline…</p> : null}
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
