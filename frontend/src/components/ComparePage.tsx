import type { ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useReport, useRuns } from "../api/queries";
import type { Report } from "../api/types";
import AppShell from "./AppShell";

const SCORE_METRICS: Array<{
  key: "accuracy" | "coverage" | "credibility" | "validity";
  label: string;
}> = [
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

function Delta({
  a,
  b,
  asPct,
  valenced = true
}: {
  a: number | null;
  b: number | null;
  asPct: boolean;
  // Counts (total/supported/contradicted/unverifiable) have no inherent
  // good/bad direction — coloring their deltas green/red implied a judgment
  // the numbers don't carry. Only the four scores keep valence colors.
  valenced?: boolean;
}) {
  if (a === null || b === null) {
    return <span className="compare__delta compare__delta--flat">—</span>;
  }
  // Round each side first so the delta always equals the difference of the two
  // displayed cells (never off-by-one against them).
  const raw = asPct ? Math.round(b * 100) - Math.round(a * 100) : b - a;
  const cls = !valenced ? "flat" : raw > 0 ? "up" : raw < 0 ? "down" : "flat";
  const sign = raw > 0 ? "+" : "";
  return (
    <span className={`compare__delta compare__delta--${cls}`}>
      {sign}
      {raw}
      {asPct ? "%" : ""}
    </span>
  );
}

// Because v2 retains every run, two can be diffed side by side. Scores are
// 0–1 fractions (or null before scoring); stats are integer counts.
const ComparePage = () => {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const a = params.get("a") ?? undefined;
  const b = params.get("b") ?? undefined;

  const runsQuery = useRuns();
  const reportA = useReport(a);
  const reportB = useReport(b);

  const titleOf = (id: string | undefined): string => {
    if (!id) return "—";
    return runsQuery.data?.find((run) => run.id === id)?.title ?? `Run ${id.slice(0, 8)}`;
  };

  const missing = !a || !b;
  const loading = reportA.isLoading || reportB.isLoading;
  const failed = reportA.error || reportB.error;
  const ra = reportA.data;
  const rb = reportB.data;

  return (
    <AppShell
      title={
        <>
          <button
            type="button"
            className="back-btn"
            onClick={() => navigate("/")}
            aria-label="Back to all runs"
          >
            ←
          </button>
          <h1>Compare runs</h1>
        </>
      }
    >
      <div className="compare-body">
        <div className="compare-inner">
          {missing ? (
            <p className="muted">Pick two runs from the gallery to compare.</p>
          ) : loading ? (
            <p className="muted">Loading runs…</p>
          ) : failed ? (
            <p className="error-text">
              {(reportA.error ?? reportB.error) instanceof Error
                ? (reportA.error ?? reportB.error)!.message
                : "One of the runs could not be loaded."}
            </p>
          ) : ra && rb ? (
            <div className="compare-card">
              <div className="compare-grid">
                <span className="compare__metric">Metric</span>
                <span className="compare-col-head">{titleOf(a)}</span>
                <span className="compare-col-head">{titleOf(b)}</span>
                <span>Δ</span>
                {SCORE_METRICS.map((metric) => (
                  <FragmentRow key={metric.key}>
                    <span className="compare__metric">{metric.label}</span>
                    <span>{fmtPct(ra.scores?.[metric.key] ?? null)}</span>
                    <span>{fmtPct(rb.scores?.[metric.key] ?? null)}</span>
                    <Delta
                      a={ra.scores?.[metric.key] ?? null}
                      b={rb.scores?.[metric.key] ?? null}
                      asPct
                    />
                  </FragmentRow>
                ))}
                {STAT_METRICS.map((metric) => (
                  <FragmentRow key={metric.key}>
                    <span className="compare__metric">{metric.label}</span>
                    <span>{ra.stats[metric.key]}</span>
                    <span>{rb.stats[metric.key]}</span>
                    <Delta
                      a={ra.stats[metric.key]}
                      b={rb.stats[metric.key]}
                      asPct={false}
                      valenced={false}
                    />
                  </FragmentRow>
                ))}
              </div>
              {ra.scores === null || rb.scores === null ? (
                <p className="muted" style={{ marginBottom: 0 }}>
                  One of these runs is not scored yet — its score cells show “—”.
                </p>
              ) : null}
              <p className="panel-note">
                Accuracy is report-position agreement: contradicted claims the report itself
                disavows count as correct, so a run can show contradicted claims and high
                accuracy at once.
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
};

// Grid cells must be direct children of the grid, so rows are fragments.
function FragmentRow({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

export default ComparePage;
