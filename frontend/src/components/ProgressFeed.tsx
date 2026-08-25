import type { JobProgressStep } from "../api/types";
import { PIPELINE_STEPS, STEP_PENDING_DETAIL, STEP_TITLES } from "../lib/steps";

// The pipeline feed. Titles are the client vocabulary; a finished step's
// detail line is the server's result string ("Extracted 37 claims"), which
// is already product voice.
export default function ProgressFeed({
  progress,
  heading = true,
  note = true
}: {
  progress: JobProgressStep[];
  heading?: boolean;
  note?: boolean;
}) {
  const byStep = Object.fromEntries(progress.map((entry) => [entry.step, entry]));
  return (
    <div className="progress-feed">
      {heading ? <h3>Verifying this report against its sources</h3> : null}
      {PIPELINE_STEPS.map((step) => {
        const entry = byStep[step];
        const status = entry?.status ?? "pending";
        const detail =
          status === "done" ? entry?.label : status === "failed" ? null : STEP_PENDING_DETAIL[step];
        return (
          <div key={step} className={`step step--${status}`}>
            <div className="step__glyph">
              {status === "done" ? (
                "✓"
              ) : status === "running" ? (
                <span className="spinner" aria-hidden>
                  ⟳
                </span>
              ) : status === "failed" ? (
                "✗"
              ) : (
                "·"
              )}
            </div>
            <div className="step__text">
              <div className="step__name">{STEP_TITLES[step] ?? step}</div>
              {detail ? <div className="step__detail">{detail}</div> : null}
            </div>
          </div>
        );
      })}
      {note ? (
        <p className="progress-note">
          You can close this page — verification continues in the background.
        </p>
      ) : null}
    </div>
  );
}
