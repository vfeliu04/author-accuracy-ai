import { ChangeEvent, FormEvent, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useReportData } from "../context/ReportDataContext";

// UploadPage lets the user add sources and the primary report before entering the dashboard.
const UploadPage = () => {
  const { internalSources, addInternalSource, reportDocument, setReportDocument } = useReportData();
  const navigate = useNavigate();

  const sourceInputRef = useRef<HTMLInputElement>(null);
  const reportInputRef = useRef<HTMLInputElement>(null);

  const handleSourceAdd = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      addInternalSource(file);
    }
    event.target.value = "";
  };

  const handleReportAdd = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setReportDocument(file);
    }
    event.target.value = "";
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    navigate("/dashboard");
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
                <span key={source.id} className="upload__item">
                  {source.name}
                </span>
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
            <button type="submit" className="upload__submit">
              ➤
            </button>
          </section>
        </div>
      </form>
    </div>
  );
};

export default UploadPage;
