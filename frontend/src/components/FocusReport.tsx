import { useSearchParams } from "react-router-dom";
import type { Report } from "../api/types";
import PdfPane from "./PdfPane";

// Full-width reader for the report PDF itself.
export default function FocusReport({ report, runId }: { report: Report; runId: string }) {
  const [, setSearchParams] = useSearchParams();
  return (
    <main className="panel panel--main">
      <div className="claims-toolbar">
        <button
          type="button"
          className="btn btn--ghost btn--small"
          onClick={() => setSearchParams({})}
        >
          ✕ Close
        </button>
        <span className="muted">The report under review</span>
      </div>
      <div className="pdf-duo">
        <div className="pdf-pane pdf-pane--single">
          <div className="pdf-pane__frame">
            <PdfPane runId={runId} docId={report.report_doc_id} title="report" />
          </div>
        </div>
      </div>
    </main>
  );
}
