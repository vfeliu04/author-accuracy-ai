import { useDocumentBlob } from "../api/queries";

// An uploaded image source, shown whole and fitted to the pane's width (a tall
// infographic scrolls). The bytes are fetched with the API key as a blob, like
// a PDF's. A verdict's quote for an image was checked against the caption
// model's reading of it, not the pixels, and the pane says so.
export default function ImagePane({
  runId,
  docId,
  title
}: {
  runId: string;
  docId: string;
  title: string | null;
}) {
  const { url, isLoading, error } = useDocumentBlob(runId, docId);

  if (isLoading) {
    return <div className="pdf-pane__empty">Loading the image…</div>;
  }
  if (error || !url) {
    return (
      <div className="pdf-pane__empty">
        Could not load the image: {error?.message ?? "unavailable"}
      </div>
    );
  }
  return (
    <figure className="image-pane">
      <div className="image-pane__canvas">
        <img className="image-pane__img" src={url} alt={`Source image: ${title ?? "untitled"}`} />
      </div>
      <figcaption className="image-pane__note">
        <span>
          Quotes from this image are checked against the model&apos;s reading of it, not against
          the image itself.
        </span>
        <a className="open-original" href={url} target="_blank" rel="noopener noreferrer">
          Open full size ↗
        </a>
      </figcaption>
    </figure>
  );
}
