import type { JobProgressStep, Report, ReportSource, RunUpload, SourceType } from "../api/types";
import { humanizeError, linksNamedIn } from "../lib/errors";
import { linkHostPath, sourceName } from "../lib/links";
import { scoreBand } from "../lib/score";

export const TIER_LABELS: Record<string, string> = {
  VERIFIED_DOI: "verified DOI",
  VERIFIED_TITLE: "verified title",
  VERIFIED_ISBN: "verified ISBN",
  // A registry record matched this source's details, but not the address it
  // was fetched from: real evidence, short of confirming this IS that work.
  MATCHED_RECORD: "registry match only",
  METADATA_ONLY: "metadata only",
  NONE: "unverified"
};

export function tierLabel(tier: string | null): string {
  return tier ? TIER_LABELS[tier] ?? tier.toLowerCase() : "";
}

// How each kind of source is marked in a list, and what a screen reader hears.
export const SOURCE_KINDS: Record<SourceType, { glyph: string; label: string }> = {
  pdf: { glyph: "📘", label: "PDF" },
  web: { glyph: "🔗", label: "Web page" },
  image: { glyph: "🖼️", label: "Image" },
  youtube: { glyph: "🎬", label: "Video" }
};

// A type this build does not know is marked as a PDF.
export function sourceKind(type: SourceType): { glyph: string; label: string } {
  return SOURCE_KINDS[type] ?? SOURCE_KINDS.pdf;
}

// Whether a source carries a credibility number.
export function isScored(source: ReportSource): source is ReportSource & { total: number } {
  return source.scorable && source.total !== null;
}

function SourceGlyph({ type }: { type: SourceType }) {
  const kind = sourceKind(type);
  return (
    <span className="src-row__icon" role="img" aria-label={kind.label}>
      {kind.glyph}
    </span>
  );
}

// An uploaded file has arrived the moment the run exists. A link's page is
// read during the first step, so until that step finishes the link is only
// queued; if the step fails, the link the error names is flagged.
function UploadStatus({
  upload,
  ingestStatus,
  failed,
  runError
}: {
  upload: RunUpload;
  ingestStatus: JobProgressStep["status"] | undefined;
  failed: boolean;
  runError: string | null;
}) {
  if (upload.url === null || ingestStatus === "done") {
    return (
      <span className="src-status src-status--ok" title="Received">
        ✓
      </span>
    );
  }
  if (ingestStatus === "failed") {
    return failed ? (
      <span className="src-status src-status--failed" title={humanizeError(runError) ?? undefined}>
        {"Couldn't open"}
      </span>
    ) : null;
  }
  return <span className="src-status src-status--queued">Queued</span>;
}

// Where a finished source stands: its verification tier when scored, or why
// it carries no score.
export function standing(source: ReportSource): string {
  if (!source.scorable) return "Not scorable";
  if (source.total === null) return "Not scored";
  return tierLabel(source.tier);
}

// A page the reader cap cut: the run was scored against its head only, so the
// row says so in plain words and keeps the numbers for the hover. Without this
// a partly-read page looks exactly like a whole one.
function PartialNote({ truncated }: { truncated: ReportSource["truncated"] }) {
  if (!truncated) return null;
  const whole = truncated.kept_chars + truncated.dropped_chars;
  return (
    <div
      className="src-row__sub src-row__sub--partial"
      title={
        `Read ${truncated.kept_chars.toLocaleString()} of ${whole.toLocaleString()} characters. ` +
        "The rest of this page was not analysed."
      }
    >
      Read in part
    </div>
  );
}

// Left panel: the report pinned on top, sources below. Before the run is
// DONE the rows are the uploads (files and links); once it is DONE they are
// every source document, with a credibility badge on each scored one.
export default function SourcesPanel({
  uploads,
  report,
  ingestStatus,
  runError = null,
  onOpenSource
}: {
  uploads: RunUpload[];
  report: Report | undefined;
  ingestStatus?: JobProgressStep["status"];
  runError?: string | null;
  onOpenSource?: (docId: string) => void;
}) {
  const documents = report?.sources ?? [];
  const showDocuments = report?.status === "DONE" && documents.length > 0;
  const reportUpload = uploads.find((upload) => upload.kind === "REPORT");
  const sourceUploads = uploads.filter((upload) => upload.kind === "SOURCE");
  const claimCount = report?.stats.claims_total ?? 0;
  // The links a failed read names, found among the links as added: the message
  // may quote a link, cut a long one short, or name where it redirected first.
  const failedLinks =
    ingestStatus === "failed" && runError !== null
      ? linksNamedIn(
          runError,
          sourceUploads.flatMap((upload) => (upload.url === null ? [] : [upload.url]))
        )
      : [];

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

        {showDocuments ? (
          <>
            <div className="src-group">Sources ({documents.length})</div>
            {documents.map((source) => (
              <button
                key={source.doc_id}
                type="button"
                className="src-row"
                onClick={() => onOpenSource?.(source.doc_id)}
              >
                <SourceGlyph type={source.source_type} />
                <div className="src-row__text">
                  <div className="src-row__name">
                    {sourceName(source, source.doc_id.slice(0, 8))}
                  </div>
                  {/* The address line, only when the name isn't already the address. */}
                  {source.url && sourceName(source, "") !== linkHostPath(source.url) ? (
                    <div className="src-row__sub src-row__sub--link" title={source.url}>
                      {linkHostPath(source.url)}
                    </div>
                  ) : null}
                  <div className="src-row__sub">{standing(source)}</div>
                  <PartialNote truncated={source.truncated} />
                </div>
                {isScored(source) ? (
                  <span className={`cred-badge cred-badge--${scoreBand(source.total)}`}>
                    {Math.round(source.total)}
                  </span>
                ) : null}
              </button>
            ))}
          </>
        ) : (
          <>
            <div className="src-group">Sources ({sourceUploads.length})</div>
            {sourceUploads.map((upload) => (
              <div key={upload.id} className="src-row src-row--static">
                <SourceGlyph type={upload.source_type} />
                <div className="src-row__text">
                  {upload.url !== null ? (
                    <div className="src-row__name src-row__name--link" title={upload.url}>
                      {linkHostPath(upload.url)}
                    </div>
                  ) : (
                    <div className="src-row__name">{upload.file_name}</div>
                  )}
                </div>
                <UploadStatus
                  upload={upload}
                  ingestStatus={ingestStatus}
                  failed={upload.url !== null && failedLinks.includes(upload.url)}
                  runError={runError}
                />
              </div>
            ))}
          </>
        )}
      </div>
    </aside>
  );
}
