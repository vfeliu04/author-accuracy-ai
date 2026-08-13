import type { ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useReport } from "../api/queries";
import type { Report } from "../api/types";

const SCORE_METRICS: Array<{ key: "accuracy" | "coverage" | "credibility" | "validity"; label: string }> = [
  { key: "accuracy", label: "Accuracy" },
  { key: "coverage", label: "Coverage" },
  { key: "credibility", label: "Credibility" },
  { key: "validity", label: "Validity" }
];

const STAT_METRICS: Array<{ key: keyof Report["stats"]; label: string }> = [
  { key: "claims_total", label: "Total claims" },
  { key: "claims_supported", label: "Supported" },
  { key: "claims_contradicted", label: "Contradicted" },
  { key: "claims_unverifiable", label: "Unverifiable" }
];

function fmtPct(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

function Delta({ a, b, asPct }: { a: number | null; b: number | null; asPct: boolean }) {
  if (a === null || b === null) {
    return <span className="compare__delta compare__delta--flat">—</span>;
  }
  // Round each side first so the delta always equals the difference of the two
  // displayed cells (never off-by-one against them).
  const raw = asPct ? Math.round(b * 100) - Math.round(a * 100) : b - a;
  const cls = raw > 0 ? "up" : raw < 0 ? "down" : "flat";
  const sign = raw > 0 ? "+" : "";
  return (
    <span className={`compare__delta compare__delta--${cls}`}>
      {sign}
      {raw}
      {asPct ? "%" : ""}
    </span>
  );
}

const GRID_STYLE = { display: "grid", gridTemplateColumns: "1.4fr 1fr 1fr 0.9fr", gap: "0.5rem 1rem", alignItems: "center" } as const;

// Because v2 retains every run, two can be diffed side by side. Scores are
// 0–1 fractions (or null before scoring); stats are integer counts.
const ComparePage = () => {
  const [params] = useSearchParams();
  const a = params.get("a") ?? undefined;
  const b = params.get("b") ?? undefined;

  const reportA = useReport(a);
  const reportB = useReport(b);

  const missing = !a || !b;
  const loading = reportA.isLoading || reportB.isLoading;
  const failed = reportA.error || reportB.error;
  const ra = reportA.data;
  const rb = reportB.data;

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>Compare runs</h1>
          <p className="dashboard__subtitle">
            {a ? a.slice(0, 8) : "—"} vs {b ? b.slice(0, 8) : "—"}
          </p>
        </div>
        <div className="dashboard__meta">
          <Link to="/runs" className="dashboard__refresh-button">
            ← All runs
          </Link>
        </div>
      </header>
      <main className="dashboard__content">
        <section className="dashboard__column" style={{ width: "100%", gridColumn: "1 / -1" }}>
          {missing ? (
            <p className="dashboard__status">Pick two runs from the history to compare.</p>
          ) : loading ? (
            <p className="dashboard__status">Loading runs…</p>
          ) : failed ? (
            <p className="dashboard__status dashboard__status--error">
              {(reportA.error ?? reportB.error) instanceof Error
                ? (reportA.error ?? reportB.error)!.message
                : "One of the runs could not be loaded."}
            </p>
          ) : ra && rb ? (
            <article className="card">
              <header className="card__header">
                <h2>Scores</h2>
              </header>
              <div style={GRID_STYLE}>
                <span className="compare__metric">Metric</span>
                <span>{a?.slice(0, 8)}</span>
                <span>{b?.slice(0, 8)}</span>
                <span>Δ</span>
                {SCORE_METRICS.map((metric) => (
                  <FragmentRow key={metric.key}>
                    <span className="compare__metric">{metric.label}</span>
                    <span>{fmtPct(ra.scores?.[metric.key] ?? null)}</span>
                    <span>{fmtPct(rb.scores?.[metric.key] ?? null)}</span>
                    <Delta a={ra.scores?.[metric.key] ?? null} b={rb.scores?.[metric.key] ?? null} asPct />
                  </FragmentRow>
                ))}
                {STAT_METRICS.map((metric) => (
                  <FragmentRow key={metric.key}>
                    <span className="compare__metric">{metric.label}</span>
                    <span>{ra.stats[metric.key]}</span>
                    <span>{rb.stats[metric.key]}</span>
                    <Delta a={ra.stats[metric.key]} b={rb.stats[metric.key]} asPct={false} />
                  </FragmentRow>
                ))}
              </div>
              {ra.scores === null || rb.scores === null ? (
                <p className="dashboard__status">
                  One of these runs is not scored yet — its score cells show “—”.
                </p>
              ) : null}
              <p style={{ margin: "0.5rem 0 0", fontSize: "0.8em", opacity: 0.7 }}>
                Accuracy is report-position agreement: contradicted claims the report itself
                disavows count as correct, so a run can show contradicted claims and high
                accuracy at once.
              </p>
            </article>
          ) : null}
        </section>
      </main>
    </div>
  );
};

// Grid cells must be direct children of the grid, so rows are fragments.
function FragmentRow({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

export default ComparePage;
