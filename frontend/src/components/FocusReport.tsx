import type { Report } from "../api/types";
import FocusToolbar from "./FocusToolbar";
import PdfPane from "./PdfPane";

// Full-width reader for the report PDF itself.
export default function FocusReport({ report, runId }: { report: Report; runId: string }) {
  return (
    <main className="panel panel--main">
      <FocusToolbar>
        <span className="muted">The report under review</span>
      </FocusToolbar>
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
