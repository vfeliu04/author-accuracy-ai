import { Link, useParams } from "react-router-dom";
import PdfPane from "./PdfPane";
import { useReport } from "../api/queries";

// The report PDF, full-height, on the v2 file endpoint.
const ReportDetail = () => {
  const { runId } = useParams<{ runId: string }>();
  const { data: report, isLoading, error } = useReport(runId);

  return (
    <div className="workspace">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>Report</h1>
          <p className="dashboard__subtitle">Run {runId?.slice(0, 8)}</p>
        </div>
        <div className="dashboard__meta">
          <Link to={`/runs/${runId}`} className="dashboard__refresh-button">
            ← Back to run
          </Link>
        </div>
      </header>
      <div className="workspace__panes">
        <section className="workspace__pane" style={{ flex: 1 }}>
          <div className="workspace__pane-body">
            {isLoading ? (
              <div className="pdf-pane pdf-pane--empty">Loading…</div>
            ) : error || !report ? (
              <div className="pdf-pane pdf-pane--empty">
                {error instanceof Error ? error.message : "Report unavailable."}
              </div>
            ) : (
              <PdfPane runId={runId as string} docId={report.report_doc_id} title="report" />
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default ReportDetail;
