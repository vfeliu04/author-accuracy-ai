import { BAND_COLORS, scoreBand } from "../lib/score";

// The animated SVG score donut (extracted from the old RatingPanel so every
// surface renders scores identically). `value` is a 0–1 fraction or null.
function ringColor(pct: number | null): string {
  if (pct === null) return "var(--color-border)";
  return BAND_COLORS[scoreBand(pct)];
}

export default function ScoreRing({
  value,
  label,
  size = 76
}: {
  value: number | null;
  label: string;
  size?: number;
}) {
  const pct = value === null ? null : Math.round(value * 100);
  const r = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const filled = pct === null ? 0 : (Math.min(pct, 100) / 100) * circ;
  const color = ringColor(pct);
  return (
    <div className="ring">
      <div
        className="ring__donut"
        style={{
          position: "relative",
          width: size,
          height: size,
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center"
        }}
      >
        <svg
          width={size}
          height={size}
          style={{ position: "absolute", top: 0, left: 0, transform: "rotate(-90deg)" }}
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="var(--color-border)"
            strokeWidth={7}
          />
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
      <span className="ring__label">{label}</span>
    </div>
  );
}
