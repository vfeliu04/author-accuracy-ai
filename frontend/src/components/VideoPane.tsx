import { useMemo } from "react";
import { useSnapshot } from "../api/queries";
import type { PageSection, VideoProvenance } from "../api/types";
import { UnreadablePageError } from "../api/v2";
import { safeHttpUrl } from "../lib/links";
import { locateQuote } from "../lib/quote";
import { BlockText, splitBlocks } from "../lib/textBlocks";

// Which caption window a citation points at: the one whose start matches
// exactly, else the one spanning that moment, else the one named by its
// title — and, failing all three (no citation, or the video's snapshot no
// longer names one), the first window, so the pane always shows something
// rather than nothing.
function pickSection(
  sections: PageSection[],
  startSeconds: number | null,
  sectionTitle: string | null
): number | null {
  if (startSeconds !== null) {
    const exact = sections.findIndex((section) => section.start_seconds === startSeconds);
    if (exact !== -1) return exact;
    const containing = sections.findIndex(
      (section) =>
        section.start_seconds !== undefined &&
        section.start_seconds <= startSeconds &&
        (section.end_seconds === undefined || startSeconds < section.end_seconds)
    );
    if (containing !== -1) return containing;
  }
  if (sectionTitle !== null) {
    const named = sections.findIndex((section) => section.title === sectionTitle);
    if (named !== -1) return named;
  }
  return sections.length > 0 ? 0 : null;
}

// A BCP-47 tag's English name ("en" -> "English"), or the tag itself when it
// can't be read (an unusual subtag Intl doesn't know).
function languageLabel(tag: string): string {
  try {
    return new Intl.DisplayNames(["en"], { type: "language" }).of(tag) ?? tag;
  } catch {
    return tag;
  }
}

// Plain words for what a quote from this video was checked against: YouTube's
// own speech recognition, or captions someone wrote.
function captionsLabel(captions: VideoProvenance["captions"]): string {
  const language = languageLabel(captions.language);
  return captions.kind === "automatic"
    ? `Automatic captions (${language})`
    : `Captions (${language})`;
}

// A YouTube video source: the player on top (or a note when the channel has
// turned off embedding), the video's own title and captions kind below, and
// the cited caption window with the quote marked.
export default function VideoPane({
  runId,
  docId,
  startSeconds,
  section,
  quote
}: {
  runId: string;
  docId: string;
  startSeconds: number | null;
  section: string | null;
  quote: string | null;
}) {
  const { data: page, error, isLoading } = useSnapshot(runId, docId);

  const sectionIndex = useMemo(
    () => (page ? pickSection(page.document.sections, startSeconds, section) : null),
    [page, startSeconds, section]
  );

  if (isLoading) {
    return <div className="pdf-pane__empty">Loading video…</div>;
  }
  if (error || !page) {
    return (
      <div className="pdf-pane__empty">
        {error instanceof UnreadablePageError
          ? error.message
          : `Could not load this video: ${error?.message ?? "unavailable"}`}
      </div>
    );
  }
  const { provenance, document: content } = page;
  const video = provenance.video;
  if (!video) {
    return <div className="pdf-pane__empty">Could not load this video: unavailable</div>;
  }
  const title = content.title || provenance.title || "Untitled video";
  // Omitted at 0/null: a fresh embed already starts at the beginning.
  const embedStart = startSeconds !== null && startSeconds > 0 ? Math.floor(startSeconds) : null;
  const embedSrc = safeHttpUrl(
    `https://www.youtube-nocookie.com/embed/${video.id}` +
      (embedStart !== null ? `?start=${embedStart}` : "")
  );
  const watchSeconds = startSeconds !== null ? Math.floor(startSeconds) : null;
  const openHref = safeHttpUrl(
    `https://www.youtube.com/watch?v=${video.id}` +
      (watchSeconds !== null ? `&t=${watchSeconds}s` : "")
  );
  const excerpt = sectionIndex !== null ? content.sections[sectionIndex] : null;
  const mark = excerpt && quote ? locateQuote(excerpt.text, quote) : null;

  return (
    <div className="video-pane">
      <div className="video-pane__player">
        {video.embeddable && embedSrc ? (
          <iframe
            key={`${video.id}-${embedStart ?? 0}`}
            className="video-pane__frame"
            src={embedSrc}
            title={title}
            referrerPolicy="strict-origin-when-cross-origin"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            sandbox="allow-scripts allow-same-origin allow-presentation"
            allowFullScreen
            loading="lazy"
          />
        ) : (
          <div className="video-pane__unplayable">
            This channel doesn&apos;t allow playing this video here.
          </div>
        )}
      </div>
      <div className="video-pane__body">
        <header className="readable__head">
          <h3 className="readable__title">{title}</h3>
          <div className="readable__meta">
            {provenance.publisher ? <span>{provenance.publisher}</span> : null}
            {provenance.publication_date ? <span>{provenance.publication_date}</span> : null}
            <span>{captionsLabel(video.captions)}</span>
            {openHref ? (
              <a
                className="readable__open"
                href={openHref}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open on YouTube ↗
              </a>
            ) : null}
          </div>
        </header>
        <div className="readable__body">
          {excerpt ? (
            <article className="readable__page">
              <h4 className="readable__heading">{excerpt.title}</h4>
              {splitBlocks(excerpt.text).map((block) => (
                <p key={block.start} className="readable__para">
                  <BlockText text={excerpt.text} block={block} mark={mark} />
                </p>
              ))}
            </article>
          ) : (
            <p className="muted">This video has no readable captions.</p>
          )}
        </div>
      </div>
    </div>
  );
}
