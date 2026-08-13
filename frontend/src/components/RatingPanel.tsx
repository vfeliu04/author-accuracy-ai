import type { Scores } from "../api/types";

function ringColor(pct: number | null): string {
  if (pct === null) return "var(--color-border)";
  return pct >= 70 ? "var(--color-success)" : pct >= 40 ? "var(--color-warning)" : "var(--color-danger)";
}

function ScoreRing({ value, label, size = 64 }: { value: number | null; label: string; size?: number }) {
  const pct = value === null ? null : Math.round(value * 100);
  const r = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const filled = pct === null ? 0 : (Math.min(pct, 100) / 100) * circ;
  const color = ringColor(pct);
  return (
    <div className="rating__card">
      <div style={{ position: "relative", width: size, height: size, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
        <svg width={size} height={size} style={{ position: "absolute", top: 0, left: 0, transform: "rotate(-90deg)" }}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-border)" strokeWidth={7} />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={7}
            strokeDasharray={`${circ}`}
            strokeDashoffset={circ - filled}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 0.6s ease" }}
          />
        </svg>
        <span style={{ fontWeight: 700, fontSize: size * 0.24, color, lineHeight: 1 }}>
          {pct === null ? "—" : pct}
        </span>
      </div>
      <span className="rating__card-label">{label}</span>
    </div>
  );
}

// The three headline scores plus coverage — each a 0–1 fraction (or null). No
// composite "overall": v2 reports accuracy, credibility, and validity as
// independent numbers on purpose. `scores` is null until the run is scored.
const METRICS: Array<{ key: keyof NonNullable<Scores>; label: string }> = [
  { key: "accuracy", label: "Accuracy" },
  { key: "coverage", label: "Coverage" },
  { key: "credibility", label: "Credibility" },
  { key: "validity", label: "Validity" }
];

const RatingPanel = ({ scores }: { scores: Scores }) => {
  return (
    <article className="card">
      <header className="card__header">
        <h2>Rating</h2>
      </header>
      {scores === null ? (
        <p className="dashboard__status">Scoring in progress…</p>
      ) : (
        <div className="rating__cards">
          {METRICS.map((metric) => (
            <ScoreRing key={metric.key} label={metric.label} value={scores[metric.key]} />
          ))}
        </div>
      )}
    </article>
  );
};

export default RatingPanel;
