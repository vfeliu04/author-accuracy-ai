import type { EvidenceSource } from "../api/types";
import { safeHttpUrl } from "../lib/links";
import PdfPane from "./PdfPane";
import ReadablePane from "./ReadablePane";

// The evidence side of a claim, chosen by what the source IS: a PDF opens at
// its cited page, a web page opens as readable text with the quote marked,
// and a source without a preview offers its original instead.
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
  const href = safeHttpUrl(source.url);
  const noun =
    source.source_type === "youtube" ? "video" : source.source_type === "image" ? "image" : "source";
  return (
    <div className="pdf-pane__empty">
      <div className="no-preview">
        <p className="no-preview__text">A preview isn&apos;t available for this {noun} yet.</p>
        {href ? (
          <a className="open-original" href={href} target="_blank" rel="noopener noreferrer">
            Open original ↗
          </a>
        ) : null}
      </div>
    </div>
  );
}
