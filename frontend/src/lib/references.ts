// The report's reference list against what the dialog already holds. Pure,
// and run on every render: the report is picked before its sources exist, so a
// verdict taken when the list was scanned would be stale for the exact case
// the checklist is there for. The server says what the report cites and where
// a free copy lives; this decides what is still missing.
import type { ScannedReference } from "../api/types";
import { checkLink } from "./links";

export function stem(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
}

// Mirror of credibility.clean_doi: the prefixes an entry may print, then the
// shape a DOI must have. Anything else is not a DOI, whatever the model said.
const DOI_PREFIX = /^(?:https?:\/\/(?:dx\.)?doi\.org\/|doi:)\s*/i;
const DOI_SHAPE = /^10\.\d{4,9}\/\S+$/;

export function cleanDoi(doi: string): string | null {
  const candidate = doi.trim().replace(DOI_PREFIX, "");
  return DOI_SHAPE.test(candidate) ? candidate : null;
}

// A DOI in a link's path — doi.org, dx.doi.org and publisher pages
// (link.springer.com/article/10.…, onlinelibrary.wiley.com/doi/10.…) all put it
// there, sometimes with its slash encoded. Only a whole path segment counts.
const DOI_IN_PATH = /(?:^|\/)(10\.\d{4,9}\/[^\s?#]+)/;

export function doiInUrl(link: string): string | null {
  let path: string;
  try {
    path = new URL(link).pathname;
  } catch {
    return null;
  }
  try {
    path = decodeURIComponent(path);
  } catch {
    // Malformed escapes stay as the link spells them.
  }
  const match = DOI_IN_PATH.exec(path);
  return match ? match[1].replace(/\/+$/, "") : null;
}

// Mirror of credibility._normalize_title: lowercase, every run of anything but
// a letter or a digit becomes one space (underscores included), Unicode-aware.
// Composed first, because a file name from a Mac can spell "é" as "e" + accent,
// and the accent alone is not a letter.
export function normalizeTitle(title: string): string {
  return title
    .normalize("NFC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

// "Smith, J." → Smith; "J. Smith", "Jane Smith", "Smith JP" → Smith; "et al."
// is not a name. An initial is a lone letter, letters with periods ("J.-P."),
// or a run of up to three capitals ("JP").
// The "et al." rule matches the words alone: the separators before them are
// dropped by the trim and the name's own trailing strip, and a `[,\s]*` in
// front of `\b` would backtrack from every position of a long separator run
// (quadratic: seconds on a 60,000-character author). An author comes from
// the model with no length bound of its own, so one is set here before any
// pattern runs; no printed name comes near it.
const AUTHOR_MAX_CHARS = 500;
const ET_AL = /\bet\s+al\.?\s*$/iu;
const INITIAL = /^(?:\p{L}\.?(?:-\p{L}\.?)*|\p{Lu}{1,3})$/u;

export function familyNames(authors: readonly string[]): string[] {
  const names: string[] = [];
  for (const author of authors) {
    const printed = author.slice(0, AUTHOR_MAX_CHARS).replace(ET_AL, "").trim();
    if (!printed) continue;
    const comma = printed.indexOf(",");
    const family =
      comma >= 0
        ? printed.slice(0, comma)
        : (printed.split(/\s+/).filter((word) => !INITIAL.test(word)).pop() ?? "");
    const name = family.replace(/[.\s]+$/, "").trim();
    if (name) names.push(name);
  }
  return names;
}

function words(text: string): string[] {
  return normalizeTitle(text).split(" ").filter(Boolean);
}

// Whole words in sequence: "hunger report" is not in "hunger reports".
function containsWords(outer: readonly string[], inner: readonly string[]): boolean {
  return inner.length <= outer.length && ` ${outer.join(" ")} `.includes(` ${inner.join(" ")} `);
}

// Fewer words match too much: "Annual Report" names half the documents ever
// filed, and a one-word file name sits inside almost any title.
const TITLE_WORD_FLOOR = 4;

function linkByDoi(ref: ScannedReference, links: readonly string[]): string | null {
  const doi = ref.doi === null ? null : cleanDoi(ref.doi);
  if (doi === null) return null;
  const wanted = doi.toLowerCase(); // DOIs are case-insensitive
  return links.find((link) => doiInUrl(link)?.toLowerCase() === wanted) ?? null;
}

// Through the dialog's own link rules, so "the same link" means what the Add
// button means: a candidate the dialog would take on its own, but refuses
// against one added link, is that link.
function linkByUrl(ref: ScannedReference, links: readonly string[]): string | null {
  for (const candidate of [ref.url, ref.suggested_url]) {
    if (!candidate || "error" in checkLink(candidate, [])) continue;
    const hit = links.find((link) => "error" in checkLink(candidate, [link]));
    if (hit) return hit;
  }
  return null;
}

function fileByTitle(ref: ScannedReference, sourceNames: readonly string[]): string | null {
  if (ref.title === null) return null;
  const title = words(ref.title);
  return (
    sourceNames.find((name) => {
      const file = words(stem(name));
      return (
        (title.length >= TITLE_WORD_FLOOR && containsWords(file, title)) ||
        (file.length >= TITLE_WORD_FLOOR && containsWords(title, file))
      );
    }) ?? null
  );
}

// Letter runs and digit runs, so "Smith2020" reads as Smith and 2020.
function runs(text: string): string[] {
  return text.normalize("NFC").toLowerCase().match(/\p{L}+|\p{N}+/gu) ?? [];
}

// Single letters are initials that slipped through, never a family name.
const NAME_LENGTH_FLOOR = 2;

function fileByAuthorYear(ref: ScannedReference, sourceNames: readonly string[]): string | null {
  if (ref.year === null) return null;
  const year = String(ref.year);
  const names = familyNames(ref.authors)
    .map(runs)
    .filter((parts) => parts.join("").length >= NAME_LENGTH_FLOOR);
  if (names.length === 0) return null;
  return (
    sourceNames.find((name) => {
      const parts = runs(stem(name));
      return (
        parts.includes(year) &&
        names.some((family) => containsWords(parts, family) || parts.includes(family.join("")))
      );
    }) ?? null
  );
}

// The source (a link, or a file name) that already stands for the cited work,
// or null when none does. Four rules, first hit wins: a link carrying the DOI;
// a link to the printed or suggested address; a file named after the title;
// a file named after an author and the year.
export function alreadyAdded(
  ref: ScannedReference,
  sourceNames: readonly string[],
  links: readonly string[]
): string | null {
  return (
    linkByDoi(ref, links) ??
    linkByUrl(ref, links) ??
    fileByTitle(ref, sourceNames) ??
    fileByAuthorYear(ref, sourceNames)
  );
}
