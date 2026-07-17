import { useNavigate } from "react-router-dom";
import type { RecommendedSource } from "../api/client";

type AnalyticsPanelProps = {
  recommendedSources: RecommendedSource[];
};

const AnalyticsPanel = ({ recommendedSources }: AnalyticsPanelProps) => {
  const navigate = useNavigate();
  const hasRecommendations = recommendedSources.length > 0;

  return (
    <article className="card">
      <header className="card__header">
        <h2>Recommended Sources</h2>
      </header>
      <div className="analytics__sources analytics__sources--stacked recommended-list">
        {hasRecommendations ? (
          <ul>
            {recommendedSources.map((source, index) => (
              <li key={source.id ?? source.title}>
                <button
                  type="button"
                  className="rec-source__card"
                  onClick={() => navigate(`/dashboard/recommendations/${index}`)}
                >
                  <span className="rec-source__title">{source.title}</span>
                  <div className="rec-source__meta">
                    {source.publication_year && (
                      <span className="rec-source__pill">{source.publication_year}</span>
                    )}
                    {source.cited_by_count != null && source.cited_by_count > 0 && (
                      <span className="rec-source__pill rec-source__pill--cited">
                        {source.cited_by_count.toLocaleString()} citations
                      </span>
                    )}
                    {source.host_venue && (
                      <span className="rec-source__venue">{source.host_venue}</span>
                    )}
                  </div>
                  {source.reason && (
                    <span className="rec-source__reason">{source.reason}</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="card__body-text analytics__source-empty">
            Run the verification pipeline to receive tailored reading recommendations that strengthen
            your report.
          </p>
        )}
      </div>
    </article>
  );
};

export default AnalyticsPanel;
