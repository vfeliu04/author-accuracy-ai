import type { Report, RunUpload } from "../api/types";

function credClass(total: number): string {
  return total >= 70 ? "cred-badge--hi" : total >= 40 ? "cred-badge--mid" : "cred-badge--lo";
}

export const TIER_LABELS: Record<string, string> = {
  VERIFIED_DOI: "verified DOI",
  VERIFIED_TITLE: "verified title",
  VERIFIED_ISBN: "verified ISBN",
  METADATA_ONLY: "metadata only",
  NONE: "unverified"
};

// Left panel: the report pinned on top, sources below. Before scoring the
// rows are the uploaded filenames; once the run is DONE they carry each
// source's credibility badge and verification tier.
export default function SourcesPanel({
  uploads,
  report,
  onOpenSource
}: {
  uploads: RunUpload[];
  report: Report | undefined;
  onOpenSource?: (docId: string) => void;
}) {
  const scored = report?.sources ?? [];
  const showScored = report?.status === "DONE" && scored.length > 0;
  const reportUpload = uploads.find((upload) => upload.kind === "REPORT");
  const sourceUploads = uploads.filter((upload) => upload.kind === "SOURCE");
  const claimCount = report?.stats.claims_total ?? 0;

  return (
    <aside className="panel panel--sources">
      <div className="panel__head">
        <h2>Sources</h2>
      </div>
      <div className="panel__body">
        <div className="src-group">Report under review</div>
        <div className="src-row src-row--report">
          <span className="src-row__icon" aria-hidden>
            📄
          </span>
          <div className="src-row__text">
            <div className="src-row__name">
              {reportUpload?.file_name ?? report?.title ?? "Report"}
            </div>
            {claimCount > 0 ? (
              <div className="src-row__sub">
                {claimCount} claim{claimCount === 1 ? "" : "s"}
              </div>
            ) : null}
          </div>
        </div>

        {showScored ? (
          <>
            <div className="src-group">Sources ({scored.length})</div>
            {scored.map((source) => (
              <button
                key={source.doc_id}
                type="button"
                className="src-row"
                onClick={() => onOpenSource?.(source.doc_id)}
              >
                <span className="src-row__icon" aria-hidden>
                  📘
                </span>
                <div className="src-row__text">
                  <div className="src-row__name">{source.title ?? source.doc_id.slice(0, 8)}</div>
                  <div className="src-row__sub">
                    {TIER_LABELS[source.tier] ?? source.tier.toLowerCase()}
                  </div>
                </div>
                <span className={`cred-badge ${credClass(source.total)}`}>
                  {Math.round(source.total)}
                </span>
              </button>
            ))}
          </>
        ) : (
          <>
            <div className="src-group">Sources ({sourceUploads.length})</div>
            {sourceUploads.map((upload) => (
              <div key={upload.id} className="src-row src-row--static">
                <span className="src-row__icon" aria-hidden>
                  📘
                </span>
                <div className="src-row__text">
                  <div className="src-row__name">{upload.file_name}</div>
                </div>
                <span className="src-status src-status--ok" title="Received">
                  ✓
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </aside>
  );
}
