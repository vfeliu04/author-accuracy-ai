import type { TextRange } from "./quote";

// Splits a source's plain text into paragraph and table blocks, and marks a
// located quote inside one — shared by every pane that shows text with a
// quote highlighted (ReadablePane's web pages, VideoPane's caption windows).

export type Block = { kind: "paragraph" | "table"; start: number; end: number };

// A section's plain text as blocks that keep their ORIGINAL offsets, so a
// located quote maps straight onto them: paragraphs split on blank lines, and
// consecutive lines starting with "|" (a table in markdown rows) kept together.
export function splitBlocks(text: string): Block[] {
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

// One block's text as plain React text, with the part the mark covers (if
// any) wrapped in <mark>. A quote spanning blocks gets one mark per block.
export function BlockText({
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
