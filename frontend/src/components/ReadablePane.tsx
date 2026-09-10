import { useEffect, useMemo, useRef } from "react";
import { useSnapshot } from "../api/queries";
import type { PageSection } from "../api/types";
import { UnreadablePageError } from "../api/v2";
import { linkHost, safeHttpUrl } from "../lib/links";
import { locateQuote, type TextRange } from "../lib/quote";

type Block = { kind: "paragraph" | "table"; start: number; end: number };

// A section's plain text as blocks that keep their ORIGINAL offsets, so a
// located quote maps straight onto them: paragraphs split on blank lines, and
// consecutive lines starting with "|" (a table in markdown rows) kept together.
function splitBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let current: Block | null = null;
  let lineStart = 0;
  for (;;) {
    const newline = text.indexOf("\n", lineStart);
    const lineEnd = newline === -1 ? text.length : newline;
    const line = text.slice(lineStart, lineEnd);
    if (line.trim() === "") {
      current = null;
    } else {
      const kind = line.trimStart().startsWith("|") ? "table" : "paragraph";
      if (current !== null && current.kind === kind) {
        current.end = lineEnd;
      } else {
        current = { kind, start: lineStart, end: lineEnd };
        blocks.push(current);
      }
    }
    if (newline === -1) return blocks;
    lineStart = newline + 1;
  }
}

type Located = { section: number; range: TextRange };

// The cited section is searched first — the same words can appear in more
// than one — then every other section in page order.
function findQuote(
  sections: PageSection[],
  quote: string | null,
  cited: string | null
): Located | null {
  if (!quote) return null;
  const indexes = sections.map((_, index) => index);
  const ordered = [
    ...indexes.filter((index) => sections[index].title === cited),
    ...indexes.filter((index) => sections[index].title !== cited)
  ];
  for (const index of ordered) {
    const range = locateQuote(sections[index].text, quote);
    if (range !== null) return { section: index, range };
  }
  return null;
}

// One block's text as plain React text, with the part the mark covers (if
// any) wrapped in <mark>. A quote spanning blocks gets one mark per block.
function BlockText({
  text,
  block,
  mark
}: {
  text: string;
  block: Block;
  mark: TextRange | null;
}) {
  if (mark === null || mark.end <= block.start || mark.start >= block.end) {
    return <>{text.slice(block.start, block.end)}</>;
  }
  const from = Math.max(mark.start, block.start);
  const to = Math.min(mark.end, block.end);
  return (
    <>
      {text.slice(block.start, from)}
      <mark>{text.slice(from, to)}</mark>
      {text.slice(to, block.end)}
    </>
  );
}

// A web page as readable text: its title and origin on top, every section
// below as plain text (never markup), the claim's quote marked and scrolled
// into view — or, when the quote can't be found, the cited section instead.
export default function ReadablePane({
  runId,
  docId,
  url,
  quote,
  section
}: {
  runId: string;
  docId: string;
  url: string | null;
  quote: string | null;
  section: string | null;
}) {
  const { data: page, error, isLoading } = useSnapshot(runId, docId);
  const bodyRef = useRef<HTMLDivElement>(null);
  const located = useMemo(
    () => (page ? findQuote(page.document.sections, quote, section) : null),
    [page, quote, section]
  );

  useEffect(() => {
    const body = bodyRef.current;
    if (!body || !page) return;
    const mark = body.querySelector("mark");
    if (mark) {
      mark.scrollIntoView?.({ block: "center" });
      return;
    }
    const cited = section ? page.document.sections.findIndex((s) => s.title === section) : -1;
    if (cited !== -1) {
      body.querySelector(`[data-section="${cited}"]`)?.scrollIntoView?.({ block: "start" });
    }
  }, [page, located, section]);

  if (isLoading) {
    return <div className="pdf-pane__empty">Loading page…</div>;
  }
  if (error || !page) {
    return (
      <div className="pdf-pane__empty">
        {error instanceof UnreadablePageError
          ? error.message
          : `Could not load this page: ${error?.message ?? "unavailable"}`}
      </div>
    );
  }

  const { provenance, document: content } = page;
  const original = provenance.final_url || url || "";
  const href = safeHttpUrl(original);
  const title = content.title || provenance.title || (href ? linkHost(href) : "Untitled page");

  return (
    <div className="readable">
      <header className="readable__head">
        <h3 className="readable__title">{title}</h3>
        <div className="readable__meta">
          {original ? (
            <span className="readable__host">{href ? linkHost(href) : original}</span>
          ) : null}
          {provenance.publisher ? <span>{provenance.publisher}</span> : null}
          {provenance.publication_date ? <span>{provenance.publication_date}</span> : null}
          {href ? (
            <a className="readable__open" href={href} target="_blank" rel="noopener noreferrer">
              Open original ↗
            </a>
          ) : null}
        </div>
      </header>
      <div className="readable__body" ref={bodyRef}>
        <article className="readable__page">
          {content.sections.map((pageSection, index) => {
            const mark = located?.section === index ? located.range : null;
            return (
              <section key={index} className="readable__section" data-section={index}>
                {pageSection.title ? (
                  <h4 className="readable__heading">{pageSection.title}</h4>
                ) : null}
                {splitBlocks(pageSection.text).map((block) =>
                  block.kind === "table" ? (
                    <pre key={block.start} className="readable__table" tabIndex={0}>
                      <BlockText text={pageSection.text} block={block} mark={mark} />
                    </pre>
                  ) : (
                    <p key={block.start} className="readable__para">
                      <BlockText text={pageSection.text} block={block} mark={mark} />
                    </p>
                  )
                )}
              </section>
            );
          })}
          {content.sections.length === 0 ? (
            <p className="muted">This page has no readable text.</p>
          ) : null}
        </article>
      </div>
    </div>
  );
}
