import { useNavigate } from "react-router-dom";
import RatingPanel from "./RatingPanel";
import SummaryPanel from "./SummaryPanel";
import { useReportData } from "../context/ReportDataContext";
import { reportScores, reportSummary } from "../data/reportData";
import { API_BASE_URL } from "../api/client";

// ReportDetail renders the full report document alongside summary and full ratings.
const ReportDetail = () => {
  const navigate = useNavigate();
  const { reportDocument, summaryData } = useReportData();

  if (!reportDocument) {
    return (
      <div className="source-detail source-detail--missing">
        <div className="source-detail__missing-card">
          <p>No report has been uploaded yet.</p>
          <button type="button" className="pill pill--filled" onClick={() => navigate("/")}>
            Return to upload
          </button>
        </div>
      </div>
    );
  }

  const fileSrc =
    reportDocument.filePath && reportDocument.filePath.startsWith("http")
      ? reportDocument.filePath
      : `${API_BASE_URL}${reportDocument.filePath ?? ""}`;

  return (
    <div className="source-detail">
      <header className="source-detail__header">
        <button
          type="button"
          className="source-detail__back-button"
          onClick={() => navigate(-1)}
          aria-label="Back to dashboard"
        >
          ←
        </button>
        <h1 className="source-detail__title">{reportDocument.name}</h1>
      </header>
      <div className="source-detail__body">
        <section className="source-detail__viewer card card--viewer">
          <iframe
            src={fileSrc}
            title={reportDocument.name}
            className="source-detail__iframe"
            loading="lazy"
          />
        </section>
        <aside className="source-detail__sidebar">
          <RatingPanel scores={summaryData?.scores ?? reportScores} showAccuracy />
          <SummaryPanel
            summary={summaryData?.report.summary ?? reportSummary}
            stats={summaryData?.stats}
            topSources={summaryData?.top_sources?.map((source) => ({
              name: source.name,
              usage_count: source.usage_count
            }))}
          />
        </aside>
      </div>
    </div>
  );
};

export default ReportDetail;
