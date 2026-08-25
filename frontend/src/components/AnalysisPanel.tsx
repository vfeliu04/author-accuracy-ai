import type { Report } from "../api/types";
import ScoreRing from "./ScoreRing";

const RING_METRICS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "credibility", label: "Credibility" },
  { key: "validity", label: "Validity" },
  { key: "coverage", label: "Coverage" }
] as const;

// Right panel: the four score rings plus tiles that open the focus modes.
// There is deliberately no composite "overall" score. The validity tile
// appears only when its focus mode exists (onOpenValidity provided).
export default function AnalysisPanel({
  report,
  onOpenClaims,
  onOpenReport,
  onOpenCredibility,
  onOpenValidity
}: {
  report: Report | undefined;
  onOpenClaims?: () => void;
  onOpenReport?: () => void;
  onOpenCredibility?: () => void;
  onOpenValidity?: () => void;
}) {
  const scores = report?.scores ?? null;
  const stats = report?.stats;
  const ready = report?.status === "DONE";

  return (
    <aside className="panel panel--analysis">
      <div className="panel__head">
        <h2>Analysis</h2>
      </div>
      <div className="panel__body">
        <div className="rings">
          {RING_METRICS.map(({ key, label }) => (
            <ScoreRing key={key} label={label} value={scores ? scores[key] : null} />
          ))}
        </div>
        {ready ? (
          <>
            <div className="tiles">
              <button type="button" className="tile tile--wide" onClick={onOpenClaims}>
                <div className="tile__row">
                  <span className="tile__name">Claims</span>
                  <span className="tile__chev">›</span>
                </div>
                {stats && stats.claims_total > 0 ? (
                  <div className="verdict-bar" aria-hidden>
                    <span className="verdict-bar__s" style={{ flex: stats.claims_supported }} />
                    <span className="verdict-bar__c" style={{ flex: stats.claims_contradicted }} />
                    <span className="verdict-bar__u" style={{ flex: stats.claims_unverifiable }} />
                  </div>
                ) : null}
                <span className="tile__sub">
                  {stats?.claims_total ?? 0} claim{stats?.claims_total === 1 ? "" : "s"}
                </span>
              </button>
              <button type="button" className="tile" onClick={onOpenReport}>
                <div className="tile__row">
                  <span className="tile__name">Report</span>
                  <span className="tile__chev">›</span>
                </div>
                <span className="tile__sub">Open the report PDF</span>
              </button>
              <button type="button" className="tile" onClick={onOpenCredibility}>
                <div className="tile__row">
                  <span className="tile__name">Credibility</span>
                  <span className="tile__chev">›</span>
                </div>
                <span className="tile__sub">Score per source</span>
              </button>
              {onOpenValidity ? (
                <button type="button" className="tile" onClick={onOpenValidity}>
                  <div className="tile__row">
                    <span className="tile__name">Validity</span>
                    <span className="tile__chev">›</span>
                  </div>
                  <span className="tile__sub">Rubric with quotes</span>
                </button>
              ) : null}
            </div>
            <p className="panel-note">
              Accuracy measures agreement with the report&rsquo;s stated positions: a claim the
              report itself disavows counts as correct when the sources contradict it.
            </p>
          </>
        ) : (
          <div className="locked-note">
            Scores, claims and chat appear here when verification completes.
          </div>
        )}
      </div>
    </aside>
  );
}
