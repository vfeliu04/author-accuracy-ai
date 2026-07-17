function ScoreRing({ score, size = 120, label }: { score: number; size?: number; label?: string }) {
  const r = (size - 16) / 2;
  const circ = 2 * Math.PI * r;
  const filled = Math.min(score, 100) / 100 * circ;
  const color = score >= 70 ? "var(--color-success)" : score >= 40 ? "var(--color-warning)" : "var(--color-danger)";
  return (
    <div style={{ position: "relative", width: size, height: size, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
      <svg width={size} height={size} style={{ position: "absolute", top: 0, left: 0, transform: "rotate(-90deg)" }}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="var(--color-border)" strokeWidth={8} />
        <circle
          cx={size/2} cy={size/2} r={r} fill="none"
          stroke={color} strokeWidth={8}
          strokeDasharray={`${circ}`}
          strokeDashoffset={circ - filled}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <span style={{
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        fontWeight: 700,
        fontSize: size * 0.22,
        color,
        lineHeight: 1,
      }}>
        {Math.round(score)}
      </span>
    </div>
  );
}

type RatingPanelProps = {
  scores: {
    overall: number;
    accuracy?: number;
    credibility: number;
    validity: number;
  };
  showAccuracy?: boolean;
};

// RatingPanel showcases the overall score bar and individual metric snapshots.
const RatingPanel = ({ scores, showAccuracy = false }: RatingPanelProps) => {
  const metricDetails = [
    ...(showAccuracy && typeof scores.accuracy === "number"
      ? [{ label: "Accuracy", value: scores.accuracy }]
      : []),
    { label: "Credibility", value: scores.credibility },
    { label: "Validity", value: scores.validity }
  ];

  return (
    <article className="card">
      <header className="card__header">
        <h2>Rating</h2>
      </header>
      <div className="rating__overall">
        <div className="rating__overall-left">
          <span className="rating__label">Overall</span>
          <span className="rating__overall-number">{(scores.overall * 100).toFixed(0)}%</span>
        </div>
        <div className="rating__bar">
          <div
            className="rating__bar-fill"
            style={{ width: `${Math.round(scores.overall * 100)}%` }}
          />
        </div>
      </div>
      <div className="rating__cards">
        {metricDetails.map((metric) => {
          const percentage = Math.round(metric.value * 100);
          const color = percentage >= 70 ? "var(--color-success)" : percentage >= 40 ? "var(--color-warning)" : "var(--color-danger)";
          return (
            <div className="rating__card" key={metric.label}>
              <ScoreRing score={percentage} size={56} />
              <span className="rating__card-label">{metric.label}</span>
              <span className="rating__card-score" style={{ color }}>{percentage}%</span>
            </div>
          );
        })}
      </div>
    </article>
  );
};

export default RatingPanel;
