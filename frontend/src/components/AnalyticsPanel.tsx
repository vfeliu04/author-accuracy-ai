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
                  className="analytics__source-button"
                  onClick={() => navigate(`/dashboard/recommendations/${index}`)}
                >
                  {source.title}
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
