import type { EvidenceSource } from "../api/types";
import { safeHttpUrl } from "../lib/links";
import ImagePane from "./ImagePane";
import PdfPane from "./PdfPane";
import ReadablePane from "./ReadablePane";
import VideoPane from "./VideoPane";

// The evidence side of a claim, chosen by what the source IS: a PDF opens at
// its cited page, a web page opens as readable text with the quote marked, an
// uploaded image is shown whole, a YouTube video opens as its player and the
// cited caption window, and a source without a preview offers its original
// instead.
export default function SourcePane({
  runId,
  source,
  quote
}: {
  runId: string;
  source: EvidenceSource;
  quote: string | null;
}) {
  if (source.source_type === "pdf") {
    return (
      <div className="pdf-pane__frame">
        <PdfPane runId={runId} docId={source.doc_id} page={source.page} title="source" />
      </div>
    );
  }
  if (source.source_type === "web") {
    return (
      <div className="pdf-pane__frame">
        <ReadablePane
          runId={runId}
          docId={source.doc_id}
          url={source.url}
          quote={quote}
          section={source.section}
        />
      </div>
    );
  }
  if (source.source_type === "image") {
    return (
      <div className="pdf-pane__frame">
        <ImagePane runId={runId} docId={source.doc_id} title={source.title} />
      </div>
    );
  }
  if (source.source_type === "youtube") {
    return (
      <div className="pdf-pane__frame">
        <VideoPane
          runId={runId}
          docId={source.doc_id}
          startSeconds={source.start_seconds}
          section={source.section}
          quote={quote}
        />
      </div>
    );
  }
  // Every SourceType is handled above; this stays as a safety net for a
  // source type this build doesn't know about.
  const href = safeHttpUrl(source.url);
  return (
    <div className="pdf-pane__empty">
      <div className="no-preview">
        <p className="no-preview__text">A preview isn&apos;t available for this source yet.</p>
        {href ? (
          <a className="open-original" href={href} target="_blank" rel="noopener noreferrer">
            Open original ↗
          </a>
        ) : null}
      </div>
    </div>
  );
}
