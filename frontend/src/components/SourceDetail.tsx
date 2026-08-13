import { Link, useParams } from "react-router-dom";

// Placeholder — the source-PDF viewer is rebuilt on the v2 file endpoint in F5.
const SourceDetail = () => {
  const { sourceId } = useParams<{ sourceId: string }>();
  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>Source {sourceId?.slice(0, 8)}</h1>
          <p className="dashboard__subtitle">The source viewer is being rebuilt.</p>
        </div>
        <div className="dashboard__meta">
          <Link to="/runs" className="dashboard__refresh-button">
            ← All runs
          </Link>
        </div>
      </header>
    </div>
  );
};

export default SourceDetail;
