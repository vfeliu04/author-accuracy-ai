import { useSearchParams } from "react-router-dom";
import type { Report } from "../api/types";

const COMPONENT_ORDER = ["coverage", "consistency", "methodology", "context", "recency"];

// Full-width validity rubric: each component's score, justification, and the
// illustrative quote (flagged when code could not find it in the report).
export default function FocusValidity({ report }: { report: Report }) {
  const [, setSearchParams] = useSearchParams();
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
      <div className="claims-toolbar">
        <button
          type="button"
          className="btn btn--ghost btn--small"
          onClick={() => setSearchParams({})}
        >
          ✕ Close
        </button>
        <span className="muted">Validity · rubric with quotes</span>
      </div>
      <div className="detail-body">
        {components && keys.length > 0 ? (
          <div className="component-grid">
            {keys.map((key) => {
              const component = components[key];
              const weight = weights?.[key];
              return (
                <div key={key} className="component-card">
                  <div className="component-card__row">
                    <span className="component-card__name">{key}</span>
                    <span className="component-card__score">
                      {component.score === null ? "—" : Math.round(component.score)}
                    </span>
                  </div>
                  {component.score === null && key === "recency" ? (
                    <p className="component-card__text">
                      No source has a known publication date — this component is excluded and
                      the other weights renormalize.
                    </p>
                  ) : null}
                  {component.justification ? (
                    <p className="component-card__text">{component.justification}</p>
                  ) : null}
                  {component.quote ? (
                    <p className="component-card__quote">
                      “{component.quote}”
                      {component.quote_verified === 0 ? (
                        <span className="quote-flag"> · quote not found in the report</span>
                      ) : null}
                    </p>
                  ) : null}
                  {weight !== undefined ? (
                    <p className="component-card__text">weight {weight}</p>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="muted">
            This run was scored before rubric details were stored — re-score it to see the
            component breakdown.
          </p>
        )}
      </div>
    </main>
  );
}
