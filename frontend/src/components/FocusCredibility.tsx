import { useSearchParams } from "react-router-dom";
import type { Report } from "../api/types";
import { BAND_COLORS, scoreBand } from "../lib/score";
import FocusToolbar from "./FocusToolbar";
import ScoreRing from "./ScoreRing";
import { TIER_LABELS } from "./SourcesPanel";

const COMPONENT_META: Array<{ key: string; label: string; max: number }> = [
  { key: "metadata_completeness", label: "Metadata completeness", max: 30 },
  { key: "authority", label: "Publisher authority", max: 30 },
  { key: "recency", label: "Recency", max: 20 },
  { key: "verification", label: "Verification", max: 20 }
];

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
              <span
                className={`cred-badge cred-badge--${scoreBand(source.total)}`}
                title="Credibility score"
              >
                {Math.round(source.total)}
              </span>
              <div>
                <div className="claim-item__text">{source.title ?? source.doc_id.slice(0, 8)}</div>
              </div>
            </button>
          ))}
          {sources.length === 0 ? <p className="muted">No sources were scored.</p> : null}
        </div>
        <div className="detail-body">
          {selected ? (
            <div className="detail-inner">
              <div className="focus-summary">
                <ScoreRing value={selected.total / 100} label="Score" size={88} />
                <div className="focus-summary__text">
                  <h3>{selected.title ?? selected.doc_id.slice(0, 8)}</h3>
                  <p>
                    {TIER_LABELS[selected.tier] ?? selected.tier}
                    {usage !== null
                      ? ` · cited by ${usage} verified verdict${usage === 1 ? "" : "s"}`
                      : ""}
                  </p>
                </div>
              </div>
              <div className="rubric-grid">
                {COMPONENT_META.map(({ key, label, max }) => {
                  const value = selected.components[key];
                  const known = value !== undefined;
                  const pct = known ? (value / max) * 100 : 0;
                  return (
                    <div key={key} className="component-card">
                      <div className="component-card__row">
                        <span className="component-card__name">{label}</span>
                        <span className="component-card__score">
                          {known ? `${Math.round(value * 10) / 10}/${max}` : "—"}
                        </span>
                      </div>
                      <div className="component-bar">
                        <span
                          style={{
                            width: `${Math.min(pct, 100)}%`,
                            background: known ? BAND_COLORS[scoreBand(pct)] : "transparent"
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <p className="muted">No sources were scored for this run.</p>
          )}
        </div>
      </div>
    </main>
  );
}
