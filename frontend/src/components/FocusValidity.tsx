import type { Report } from "../api/types";
import { BAND_COLORS, scoreBand } from "../lib/score";
import FocusToolbar from "./FocusToolbar";
import ScoreRing from "./ScoreRing";

const COMPONENT_ORDER = ["coverage", "consistency", "methodology", "context", "recency"];

// Quotes come verbatim from the parsed report, markdown markers and all —
// strip heading/emphasis noise for display only.
function displayQuote(quote: string): string {
  return quote.replace(/^[#>*\s]+/, "").trim();
}

// Full-width validity rubric: an overall summary, then each component's
// score, justification, and the illustrative quote (flagged when code could
// not find it in the report).
export default function FocusValidity({ report }: { report: Report }) {
  const detail = report.validity_detail;
  const components = detail?.components ?? null;
  const weights = detail?.weights_used ?? null;

  const keys = components
    ? [
        ...COMPONENT_ORDER.filter((key) => key in components),
        ...Object.keys(components).filter((key) => !COMPONENT_ORDER.includes(key))
      ]
    : [];

  return (
    <main className="panel panel--main">
      <FocusToolbar>
        <span className="muted">Validity · rubric with quotes</span>
      </FocusToolbar>
      <div className="detail-body">
        <div className="detail-inner">
          {components && keys.length > 0 ? (
            <>
              <div className="focus-summary">
                <ScoreRing
                  value={report.scores?.validity ?? null}
                  label="Validity"
                  size={88}
                />
                <div className="focus-summary__text">
                  <h3>How the report itself holds up</h3>
                  <p>
                    Each component is scored against the report text; illustrative quotes are
                    checked by code against the report.
                  </p>
                </div>
              </div>
              <div className="rubric-grid">
                {keys.map((key) => {
                  const component = components[key];
                  const weight = weights?.[key];
                  const known = component.score !== null;
                  return (
                    <div key={key} className="component-card">
                      <div className="component-card__row">
                        <span className="component-card__name">{key}</span>
                        <span className="component-card__score">
                          {known ? Math.round(component.score as number) : "—"}
                          {weight !== undefined ? (
                            <span className="component-card__weight"> · w {weight}</span>
                          ) : null}
                        </span>
                      </div>
                      <div className="component-bar">
                        <span
                          style={{
                            width: `${known ? Math.min(component.score as number, 100) : 0}%`,
                            background: known
                              ? BAND_COLORS[scoreBand(component.score as number)]
                              : "transparent"
                          }}
                        />
                      </div>
                      {!known && key === "recency" ? (
                        <p className="component-card__text">
                          No source has a known publication date — this component is excluded
                          and the other weights renormalize.
                        </p>
                      ) : null}
                      {component.justification ? (
                        <p className="component-card__text">{component.justification}</p>
                      ) : null}
                      {component.quote ? (
                        <p className="component-card__quote component-card__quote--clamped">
                          “{displayQuote(component.quote)}”
                          {component.quote_verified === 0 ? (
                            <span className="quote-flag"> · quote not found in the report</span>
                          ) : null}
                        </p>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <p className="muted">
              This run was scored before rubric details were stored — re-score it to see the
              component breakdown.
            </p>
          )}
        </div>
      </div>
    </main>
  );
}
