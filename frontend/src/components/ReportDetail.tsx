import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import RatingPanel from "./RatingPanel";
import SummaryPanel from "./SummaryPanel";
import { useReportData } from "../context/ReportDataContext";
import { reportScores, reportSummary } from "../data/reportData";
import { API_BASE_URL, API_KEY } from "../api/client";

// ReportDetail renders the full report document alongside summary and full ratings.
const ReportDetail = () => {
  const navigate = useNavigate();
  const { reportDocument, summaryData } = useReportData();
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  const uploadId = reportDocument?.id;

  useEffect(() => {
    if (!uploadId) return;

    let objectUrl = "";

    fetch(`${API_BASE_URL}/api/uploads/${uploadId}/file`, {
      headers: API_KEY ? { "X-API-Key": API_KEY } : {}
    })
      .then(r => {
        if (!r.ok) throw new Error("Failed to load PDF");
        return r.blob();
      })
      .then(blob => {
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      })
      .catch(() => setBlobUrl(null));

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [uploadId]);

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
            src={blobUrl ?? ""}
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
