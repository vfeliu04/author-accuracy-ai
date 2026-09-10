// Finds a verdict's quote inside a source's text so the reader sees it marked.
// The server accepted the quote after neutralizing typography (case, curly
// quotes, dashes, whitespace runs), so both sides are folded the same way here
// and the hit is mapped back to ORIGINAL character offsets for the highlight.

export type TextRange = { start: number; end: number }; // end is exclusive

const FOLD: Record<string, string> = {
  "‘": "'",
  "’": "'",
  "‚": "'",
  "‛": "'",
  "“": '"',
  "”": '"',
  "„": '"',
  "‟": '"',
  "–": "-",
  "—": "-"
};

// JavaScript's \s covers newlines, tabs, and the Unicode spaces (non-breaking,
// thin, ideographic, ...).
const WHITESPACE = /\s/;

type Folded = { text: string; starts: number[]; ends: number[] };

// Folds `input` and records, for every folded code unit, the span of the
// original character — or whole whitespace run — it came from. A lowercase
// form can be longer than its original ("İ" → "i̇"), so offsets can't be shared.
function fold(input: string): Folded {
  const units: string[] = [];
  const starts: number[] = [];
  const ends: number[] = [];
  let offset = 0;
  let inWhitespace = false;
  for (const char of input) {
    const start = offset;
    offset += char.length;
    if (WHITESPACE.test(char)) {
      if (inWhitespace) {
        ends[ends.length - 1] = offset;
      } else {
        units.push(" ");
        starts.push(start);
        ends.push(offset);
        inWhitespace = true;
      }
      continue;
    }
    inWhitespace = false;
    const folded = (FOLD[char] ?? char).toLowerCase();
    for (let i = 0; i < folded.length; i += 1) {
      units.push(folded[i]);
      starts.push(start);
      ends.push(offset);
    }
  }
  return { text: units.join(""), starts, ends };
}

// The original-text range of the first occurrence of `quote`, or null.
export function locateQuote(text: string, quote: string): TextRange | null {
  const needle = fold(quote).text.trim();
  if (needle === "") return null;
  const haystack = fold(text);
  const at = haystack.text.indexOf(needle);
  if (at === -1) return null;
  return { start: haystack.starts[at], end: haystack.ends[at + needle.length - 1] };
}
