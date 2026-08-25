import { useSearchParams } from "react-router-dom";
import type { Report } from "../api/types";
import FocusToolbar from "./FocusToolbar";
import { TIER_LABELS } from "./SourcesPanel";

const COMPONENT_META: Array<{ key: string; label: string; max: number }> = [
  { key: "metadata_completeness", label: "Metadata completeness", max: 30 },
  { key: "authority", label: "Publisher authority", max: 30 },
  { key: "recency", label: "Recency", max: 20 },
  { key: "verification", label: "Verification", max: 20 }
];

const METHOD_LINES: Record<string, string> = {
  usage_weighted_mean:
    "The run-level score weights each source by how many verified verdicts cite it.",
  unweighted_mean_no_usage:
    "No verdict cites any source, so the run-level score is a plain average.",
  no_sources: "This run has no sources."
};

// Full-width per-source credibility breakdown. The selected source can be
// deep-linked via ?source=<doc_id> (the sources panel links here).
export default function FocusCredibility({ report }: { report: Report }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const sources = report.sources;
  const selectedParam = searchParams.get("source");
  const selected = sources.find((source) => source.doc_id === selectedParam) ?? sources[0] ?? null;

  const usage =
    report.credibility_detail?.sources?.find((entry) => entry.doc_id === selected?.doc_id)
      ?.usage ?? null;
  const methodLine = report.credibility_detail?.method
    ? METHOD_LINES[report.credibility_detail.method]
    : null;

  const selectSource = (docId: string) => {
    const next = new URLSearchParams(searchParams);
    next.set("source", docId);
    setSearchParams(next, { replace: true });
  };

  return (
    <main className="panel panel--main">
      <FocusToolbar>
        <span className="muted">Credibility · score per source</span>
      </FocusToolbar>
      <div className="claims-split">
        <div className="detail-list">
          {sources.map((source) => (
            <button
              key={source.doc_id}
              type="button"
              className={`claim-item${selected?.doc_id === source.doc_id ? " selected" : ""}`}
              onClick={() => selectSource(source.doc_id)}
            >
              <div>
                <div className="claim-item__text">{source.title ?? source.doc_id.slice(0, 8)}</div>
                <div className="claim-item__cite">
                  {Math.round(source.total)}/100 · {TIER_LABELS[source.tier] ?? source.tier}
                </div>
              </div>
            </button>
          ))}
          {sources.length === 0 ? <p className="muted">No sources were scored.</p> : null}
        </div>
        <div className="detail-body">
          {selected ? (
            <>
              <div className="detail-section">
                <h3>{selected.title ?? selected.doc_id.slice(0, 8)}</h3>
                <p className="muted" style={{ margin: 0 }}>
                  {Math.round(selected.total)}/100 · {TIER_LABELS[selected.tier] ?? selected.tier}
                  {usage !== null
                    ? ` · cited by ${usage} verified verdict${usage === 1 ? "" : "s"}`
                    : ""}
                </p>
              </div>
              <div className="detail-section">
                <h4>Components</h4>
                <div className="component-grid">
                  {COMPONENT_META.map(({ key, label, max }) => (
                    <div key={key} className="component-card">
                      <div className="component-card__row">
                        <span className="component-card__name">{label}</span>
                        <span className="component-card__score">
                          {selected.components[key] !== undefined
                            ? `${Math.round(selected.components[key] * 10) / 10}/${max}`
                            : "—"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {methodLine ? <p className="muted">{methodLine}</p> : null}
            </>
          ) : (
            <p className="muted">No sources were scored for this run.</p>
          )}
        </div>
      </div>
    </main>
  );
}
