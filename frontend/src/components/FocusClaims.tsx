import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import type { Report, Verdict } from "../api/types";
import ClaimBadges from "./ClaimBadges";
import FocusToolbar from "./FocusToolbar";
import PdfPane from "./PdfPane";

const FILTERS: Array<Verdict | "ALL"> = ["ALL", "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"];

const FILTER_LABELS: Record<Verdict | "ALL", string> = {
  ALL: "All",
  SUPPORTED: "Supported",
  CONTRADICTED: "Contradicted",
  UNVERIFIABLE: "Unverifiable"
};

// Full-width claims focus: list on the left, the selected claim's report
// page and source page SIDE BY SIDE on the right. Selection and filter live
// in the URL (replace, not push) so a claim can be deep-linked; Close drops
// the params, and the browser Back button does the same.
export default function FocusClaims({ report, runId }: { report: Report; runId: string }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [search, setSearch] = useState("");

  const verdictParam = searchParams.get("verdict");
  const filter: Verdict | "ALL" = (FILTERS as string[]).includes(verdictParam ?? "")
    ? (verdictParam as Verdict)
    : "ALL";
  const selectedId = searchParams.get("claim");

  const needle = search.trim().toLowerCase();
  const claims = report.claims.filter(
    (claim) =>
      (filter === "ALL" || claim.verdict === filter) &&
      (needle === "" || claim.text.toLowerCase().includes(needle))
  );
  // Resolve the selection within the FILTERED list, so switching the filter
  // falls back to a visible claim rather than showing a now-hidden one.
  const selected = claims.find((claim) => claim.claim_id === selectedId) ?? claims[0] ?? null;

  const setParams = (updates: Record<string, string | null>) => {
    const next = new URLSearchParams(searchParams);
    for (const [key, value] of Object.entries(updates)) {
      if (value === null) next.delete(key);
      else next.set(key, value);
    }
    setSearchParams(next, { replace: true });
  };

  return (
    <main className="panel panel--main">
      <FocusToolbar>
        {FILTERS.map((option) => {
          const count =
            option === "ALL"
              ? report.claims.length
              : report.claims.filter((claim) => claim.verdict === option).length;
          return (
            <button
              key={option}
              type="button"
              className={`filter-chip${option === filter ? " active" : ""}`}
              onClick={() =>
                setParams({ verdict: option === "ALL" ? null : option, claim: null })
              }
            >
              {FILTER_LABELS[option]} {count}
            </button>
          );
        })}
        <label className="claims-search">
          <input
            placeholder="Search claims…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
      </FocusToolbar>
      <div className="claims-split">
        <div className="claims-list">
          {claims.map((claim) => (
            <button
              key={claim.claim_id}
              type="button"
              className={`claim-item${selected?.claim_id === claim.claim_id ? " selected" : ""}`}
              onClick={() => setParams({ claim: claim.claim_id })}
            >
              <span
                className={`verdict-pill verdict-pill--${claim.verdict.toLowerCase()}`}
                title={claim.verdict}
              >
                {claim.verdict[0]}
              </span>
              <div>
                <div className="claim-item__text">{claim.text}</div>
                <div className="claim-item__cite">
                  {claim.evidence_source
                    ? `${claim.evidence_source.title ?? "Source"}${
                        claim.evidence_source.page !== null
                          ? ` · p.${claim.evidence_source.page}`
                          : ""
                      }`
                    : "No source coverage"}
                </div>
              </div>
            </button>
          ))}
          {claims.length === 0 ? <p className="muted">No claims match.</p> : null}
        </div>

        <div className="claim-compare">
          {selected ? (
            <>
              <div className="pdf-duo">
                <div className="pdf-pane">
                  <div className="pdf-pane__head">
                    <span className="pdf-pane__doc">
                      <strong>Report</strong>
                      {selected.page !== null ? ` · p.${selected.page}` : ""}
                    </span>
                  </div>
                  <div className="pdf-pane__frame">
                    <PdfPane
                      runId={runId}
                      docId={report.report_doc_id}
                      page={selected.page}
                      title="report"
                    />
                  </div>
                </div>
                <div className="pdf-pane">
                  <div className="pdf-pane__head">
                    <span className="pdf-pane__doc">
                      <strong>Source</strong>
                      {selected.evidence_source
                        ? ` · ${selected.evidence_source.title ?? "untitled"}${
                            selected.evidence_source.page !== null
                              ? ` · p.${selected.evidence_source.page}`
                              : ""
                          }`
                        : ""}
                    </span>
                  </div>
                  {selected.evidence_source ? (
                    <div className="pdf-pane__frame">
                      <PdfPane
                        runId={runId}
                        docId={selected.evidence_source.doc_id}
                        page={selected.evidence_source.page}
                        title="source"
                      />
                    </div>
                  ) : (
                    <div className="pdf-pane__empty">
                      This claim has no quoted source evidence.
                    </div>
                  )}
                </div>
              </div>
              <div className="compare-head compare-head--below">
                <div className="compare-head__badges">
                  <ClaimBadges claim={selected} showYearFlag />
                </div>
                <p className="compare-head__claim">“{selected.text}”</p>
                <p className="compare-head__why">{selected.rationale}</p>
                {selected.quote ? (
                  <p className="compare-head__quote">
                    “{selected.quote}”
                    {selected.quote_verified === 1 ? (
                      <span className="muted"> · quote verified ✓</span>
                    ) : null}
                  </p>
                ) : null}
              </div>
            </>
          ) : (
            <p className="muted" style={{ padding: "1rem" }}>
              No claims match this filter.
            </p>
          )}
        </div>
      </div>
    </main>
  );
}
