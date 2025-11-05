import { useNavigate, useParams } from "react-router-dom";
import SummaryPanel from "./SummaryPanel";
import RatingPanel from "./RatingPanel";
import { useReportData } from "../context/ReportDataContext";

// SourceDetail displays a selected internal source alongside supporting context.
const SourceDetail = () => {
  const navigate = useNavigate();
  const { sourceId } = useParams();
  const { getInternalSourceById } = useReportData();

  const source = sourceId ? getInternalSourceById(sourceId) : undefined;

  if (!source) {
    return (
      <div className="source-detail source-detail--missing">
        <div className="source-detail__missing-card">
          <p>We couldn&apos;t find that source.</p>
          <button type="button" className="pill pill--filled" onClick={() => navigate("/")}>
            Return to dashboard
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
        <h1 className="source-detail__title">{source.name}</h1>
      </header>
      <div className="source-detail__body">
        <section className="source-detail__viewer card card--viewer">
          <iframe
            src={source.filePath}
            title={source.name}
            className="source-detail__iframe"
            loading="lazy"
          />
        </section>
        <aside className="source-detail__sidebar">
          {source.scores ? (
            <RatingPanel
              scores={{
                overall:
                  source.scores.overall ??
                  (source.scores.credibility + source.scores.validity) / 2,
                credibility: source.scores.credibility,
                validity: source.scores.validity
              }}
            />
          ) : (
            <article className="card card--empty">
              <p className="card__body-text">Ratings for this source are coming soon.</p>
            </article>
          )}
          <SummaryPanel
            summary={
              source.summary ??
              "Summary for this source will appear here once available."
            }
          />
        </aside>
      </div>
    </div>
  );
};

export default SourceDetail;
