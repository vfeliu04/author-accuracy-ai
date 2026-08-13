import { Link } from "react-router-dom";

// Placeholder — the report-PDF viewer is rebuilt on the v2 file endpoint in F5.
const ReportDetail = () => (
  <div className="dashboard">
    <header className="dashboard__header">
      <div className="dashboard__title-block">
        <h1>Report</h1>
        <p className="dashboard__subtitle">The report viewer is being rebuilt.</p>
      </div>
      <div className="dashboard__meta">
        <Link to="/runs" className="dashboard__refresh-button">
          ← All runs
        </Link>
      </div>
    </header>
  </div>
);

export default ReportDetail;
