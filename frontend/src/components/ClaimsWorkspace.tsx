import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import PdfPane from "./PdfPane";
import { useReport } from "../api/queries";
import type { Verdict } from "../api/types";

const VERDICT_COLOR: Record<Verdict, string> = {
  SUPPORTED: "var(--color-success)",
  CONTRADICTED: "var(--color-danger)",
  UNVERIFIABLE: "var(--color-warning)"
};

const FILTERS: Array<Verdict | "ALL"> = ["ALL", "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"];

// Three panes: the claim list, the report page the claim came from, and the
// source page its evidence was quoted from — both deep-linked via #page.
const ClaimsWorkspace = () => {
  const { runId } = useParams<{ runId: string }>();
  const { data: report, isLoading, error } = useReport(runId);
  const [filter, setFilter] = useState<Verdict | "ALL">("ALL");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  if (isLoading) {
    return <div className="dashboard"><p className="dashboard__status">Loading report…</p></div>;
  }
  if (error || !report) {
    return (
      <div className="dashboard">
        <p className="dashboard__status dashboard__status--error">
          {error instanceof Error ? error.message : "Report unavailable."}
        </p>
      </div>
    );
  }

  const claims = report.claims.filter((c) => filter === "ALL" || c.verdict === filter);
  // Resolve the selection within the FILTERED list, so switching the filter
  // falls back to a visible claim rather than showing a now-hidden one.
  const selected = claims.find((c) => c.claim_id === selectedId) ?? claims[0] ?? null;

  return (
    <div className="workspace">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>Claims Workspace</h1>
          <p className="dashboard__subtitle">{report.claims.length} claims · run {runId?.slice(0, 8)}</p>
        </div>
        <div className="dashboard__meta">
          <Link to={`/runs/${runId}`} className="dashboard__refresh-button">
            ← Back to run
          </Link>
        </div>
      </header>

      <div className="workspace__filters">
        {FILTERS.map((option) => (
          <button
            key={option}
            type="button"
            className={`chat__suggestion-chip${option === filter ? " chat__mode-option--active" : ""}`}
            onClick={() => setFilter(option)}
          >
            {option}
          </button>
        ))}
      </div>

      <div className="workspace__panes">
        <section className="workspace__list">
          {claims.map((claim) => (
            <button
              key={claim.claim_id}
              type="button"
              className={`upload__item${selected?.claim_id === claim.claim_id ? " upload__item--report" : ""}`}
              style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}
              onClick={() => setSelectedId(claim.claim_id)}
            >
              <span style={{ display: "flex", gap: 6 }}>
                <span
                  className="source-pill__badge"
                  style={{ background: VERDICT_COLOR[claim.verdict], color: "#fff" }}
                >
                  {claim.verdict}
                </span>
                {claim.stance === "disavowed" ? (
                  <span
                    className="source-pill__badge"
                    title="The report itself marks this claim false"
                  >
                    disavowed
                  </span>
                ) : null}
              </span>
              <span className="upload__item-name" style={{ whiteSpace: "normal" }}>{claim.text}</span>
            </button>
          ))}
          {claims.length === 0 ? <p className="dashboard__status">No claims match this filter.</p> : null}
        </section>

        <section className="workspace__pane">
          <div className="workspace__pane-label">Report{selected?.page ? ` · p.${selected.page}` : ""}</div>
          <div className="workspace__pane-body">
            <PdfPane runId={runId as string} docId={report.report_doc_id} page={selected?.page} title="report" />
          </div>
        </section>

        <section className="workspace__pane">
          <div className="workspace__pane-label">
            Source
            {selected?.evidence_source?.page ? ` · p.${selected.evidence_source.page}` : ""}
          </div>
          <div className="workspace__pane-body">
            {selected?.evidence_source ? (
              <PdfPane
                runId={runId as string}
                docId={selected.evidence_source.doc_id}
                page={selected.evidence_source.page}
                title="source"
              />
            ) : (
              <div className="pdf-pane pdf-pane--empty">
                This claim has no quoted source evidence.
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default ClaimsWorkspace;
