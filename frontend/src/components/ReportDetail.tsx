import { useNavigate } from "react-router-dom";
import RatingPanel from "./RatingPanel";
import SummaryPanel from "./SummaryPanel";
import { useReportData } from "../context/ReportDataContext";
import { reportScores, reportSummary } from "../data/reportData";

// ReportDetail renders the full report document alongside summary and full ratings.
const ReportDetail = () => {
  const navigate = useNavigate();
  const { reportDocument } = useReportData();

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
            src={reportDocument.filePath}
            title={reportDocument.name}
            className="source-detail__iframe"
            loading="lazy"
          />
        </section>
        <aside className="source-detail__sidebar">
          <RatingPanel scores={reportScores} showAccuracy />
          <SummaryPanel summary={reportSummary} />
        </aside>
      </div>
    </div>
  );
};

export default ReportDetail;
