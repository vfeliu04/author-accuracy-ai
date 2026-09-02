import { usePdfBlob } from "../api/queries";

// Renders a run's stored PDF in an iframe, deep-linked to a page. The bytes are
// fetched with the API key as a blob (an iframe can't send headers) and shown
// via an object URL. Keyed by page so re-selecting a claim re-scrolls the view.
const PdfPane = ({
  runId,
  docId,
  page,
  title
}: {
  runId: string;
  docId: string | null | undefined;
  page?: number | null;
  title: string;
}) => {
  const { url, isLoading, error } = usePdfBlob(runId, docId ?? undefined);

  if (!docId) {
    return <div className="pdf-pane__empty">No document for this pane.</div>;
  }
  if (isLoading) {
    return <div className="pdf-pane__empty">Loading {title}…</div>;
  }
  if (error || !url) {
    return (
      <div className="pdf-pane__empty">
        Could not load {title}: {error?.message ?? "unavailable"}
      </div>
    );
  }
  // Chromium's viewer honors these fragment options: no toolbar/side panel
  // (our pane header already names the document and page) and fit-to-width.
  // Other engines ignore what they don't support and still render the PDF.
  const options = [page ? `page=${page}` : null, "toolbar=0", "navpanes=0", "view=FitH"]
    .filter(Boolean)
    .join("&");
  const src = `${url}#${options}`;
  return (
    <iframe
      key={`${docId}-${page ?? 0}`}
      title={title}
      src={src}
      style={{ width: "100%", height: "100%", border: 0, background: "#fff" }}
    />
  );
};

export default PdfPane;
