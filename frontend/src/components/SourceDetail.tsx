import { Link, useParams } from "react-router-dom";
import PdfPane from "./PdfPane";
import { useReport } from "../api/queries";
import type { Verdict } from "../api/types";

const VERDICT_COLOR: Record<Verdict, string> = {
  SUPPORTED: "var(--color-success)",
  CONTRADICTED: "var(--color-danger)",
  UNVERIFIABLE: "var(--color-warning)"
};

// A source PDF plus the claims whose evidence was quoted from it.
const SourceDetail = () => {
  const { runId, sourceId } = useParams<{ runId: string; sourceId: string }>();
  const { data: report, isLoading, error } = useReport(runId);

  const source = report?.sources.find((s) => s.doc_id === sourceId);
  const claimsUsing = report?.claims.filter((c) => c.evidence_source?.doc_id === sourceId) ?? [];

  return (
    <div className="workspace">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>{source?.title ?? `Source ${sourceId?.slice(0, 8)}`}</h1>
          {source ? (
            <p className="dashboard__subtitle">
              tier {source.tier} · credibility {Math.round(source.total)}/100 · {claimsUsing.length}{" "}
              claim{claimsUsing.length !== 1 ? "s" : ""} cite this source
            </p>
          ) : null}
        </div>
        <div className="dashboard__meta">
          <Link to={`/runs/${runId}`} className="dashboard__refresh-button">
            ← Back to run
          </Link>
        </div>
      </header>
      <div className="workspace__panes">
        <section className="workspace__list">
          {isLoading ? <p className="dashboard__status">Loading…</p> : null}
          {error ? (
            <p className="dashboard__status dashboard__status--error">
              {error instanceof Error ? error.message : "Unavailable."}
            </p>
          ) : null}
          {claimsUsing.map((claim) => (
            <div
              key={claim.claim_id}
              className="upload__item"
              style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}
            >
              <span
                className="source-pill__badge"
                style={{ background: VERDICT_COLOR[claim.verdict], color: "#fff" }}
              >
                {claim.verdict}
              </span>
              <span className="upload__item-name" style={{ whiteSpace: "normal" }}>{claim.text}</span>
            </div>
          ))}
          {report && claimsUsing.length === 0 ? (
            <p className="dashboard__status">No claims cite this source.</p>
          ) : null}
        </section>
        <section className="workspace__pane" style={{ flex: 1 }}>
          <div className="workspace__pane-body">
            <PdfPane runId={runId as string} docId={sourceId} title="source" />
          </div>
        </section>
      </div>
    </div>
  );
};

export default SourceDetail;
