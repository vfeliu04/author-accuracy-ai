import type { Scores } from "../api/types";
import ScoreRing from "./ScoreRing";

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
        <>
          <div className="rating__cards">
            {METRICS.map((metric) => (
              <ScoreRing key={metric.key} label={metric.label} value={scores[metric.key]} />
            ))}
          </div>
          <p style={{ margin: "0.5rem 0 0", fontSize: "0.8em", opacity: 0.7 }}>
            Accuracy measures agreement with the report&rsquo;s stated positions: a claim the
            report itself disavows counts as correct when the sources contradict it.
          </p>
        </>
      )}
    </article>
  );
};

export default RatingPanel;
