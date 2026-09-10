// Raw exception text is for logs; users get the translation (the raw message
// stays visible in a collapsible block wherever this is rendered).

type ErrorHint = {
  match: RegExp;
  hint: (found: RegExpExecArray) => string;
  // A link failure names the link, so the reader knows which one to fix.
  aboutLink?: boolean;
  // "HTTP 404" and "timed out" are generic network phrasing; they describe a
  // link only when the message names one.
  needsLink?: boolean;
};

const ERROR_HINTS: ErrorHint[] = [
  {
    match: /APIConnectionError|Connection error/i,
    hint: () =>
      "The server lost its network connection mid-run (laptop sleep or dropped Wi-Fi are the usual causes). Retrying resumes where it stopped."
  },
  {
    match: /TimeoutError.*Batch|still 'in_progress'/i,
    hint: () =>
      "The verification batch was still queued on the provider's side when the app stopped waiting. The batch keeps its place — retrying reattaches to it at no extra cost."
  },
  {
    match: /credit balance|billing/i,
    hint: () => "The API account looks out of credit — top up at console.anthropic.com, then retry."
  },
  // Registry outages while scoring also carry a URL and an HTTP status, so they
  // must be recognized before the link failures below.
  {
    match: /gave no answer after/i,
    hint: () =>
      "A publication registry (Crossref or a book catalog) didn't respond while sources were being checked. Retrying picks up where it stopped."
  },
  {
    match: /private or reserved network address/i,
    aboutLink: true,
    hint: () => "That link points to a private network address, so it can't be opened."
  },
  {
    match: /unsupported content type/i,
    aboutLink: true,
    hint: () => "That link isn't a web page or PDF."
  },
  {
    match: /no readable article text/i,
    aboutLink: true,
    hint: () =>
      "That page has no readable text. Pages that need JavaScript to show their content can't be read."
  },
  {
    match: /\bHTTP (\d{3})\b/,
    aboutLink: true,
    needsLink: true,
    hint: (found) => `The site returned an error (${found[1]}).`
  },
  {
    match: /timed out|deadline/i,
    aboutLink: true,
    needsLink: true,
    hint: () => "The site took too long to respond."
  }
];

// A link in a message runs to the first space, double quote, or angle bracket.
// Parentheses, square brackets, and apostrophes can belong to a link
// (".../wiki/Mercury_(planet)", ".../it's-here"), so they are kept here and
// trimmed below only when they belong to the sentence instead.
const LINK = /https?:\/\/[^\s"<>]+/gi;

// How the server marks a long link it shortened for the message.
const CUT_MARK = "...";

type NamedLink = { link: string; cut: boolean };

function count(text: string, char: string): number {
  return text.split(char).length - 1;
}

// One matched link without what follows it in the sentence: punctuation, the
// quote around it, a bracket closing text the link never opened, and a cut
// mark come off the end until the last character belongs to the link.
function trimLink(raw: string): NamedLink {
  let link = raw;
  let cut = false;
  for (;;) {
    const last = link.slice(-1);
    if (link.endsWith(CUT_MARK)) {
      link = link.slice(0, -CUT_MARK.length);
      cut = true;
    } else if (
      /[.,;:!?]/.test(last) ||
      (last === "'" && count(link, "'") % 2 === 1) ||
      (last === ")" && count(link, "(") < count(link, ")")) ||
      (last === "]" && count(link, "[") < count(link, "]"))
    ) {
      link = link.slice(0, -1);
    } else {
      return { link, cut };
    }
  }
}

function namedLinks(error: string): NamedLink[] {
  return Array.from(error.matchAll(LINK), (found) => trimLink(found[0]));
}

// The first link a message names, without the quotes, brackets, or sentence
// punctuation around it.
export function namedLink(error: string): string | null {
  return namedLinks(error)[0]?.link ?? null;
}

// The message with every link blanked out, so words in an address
// (".../billing", ".../batch-jobs", ".../deadlines") never pick a translation.
function withoutLinks(error: string): string {
  return error.replace(LINK, (raw) => ` ${raw.slice(trimLink(raw).link.length)}`);
}

function appearsWhole(error: string, link: string, links: readonly string[]): boolean {
  if (link === "") return false;
  for (let at = error.indexOf(link); at !== -1; at = error.indexOf(link, at + 1)) {
    const longer = links.some((other) => other.length > link.length && error.startsWith(other, at));
    if (!longer) return true;
  }
  return false;
}

// Which of a run's added links a failure message names. A link counts where
// it appears whole, unless a longer added link starting the same way is what
// appears there (".../report" inside ".../report-2024"), or where the message
// cut a long link short and the added link starts with what is left.
export function linksNamedIn(error: string, links: readonly string[]): string[] {
  const cutStarts = namedLinks(error)
    .filter((named) => named.cut)
    .map((named) => named.link);
  return links.filter(
    (link) =>
      appearsWhole(error, link, links) || cutStarts.some((start) => link.startsWith(start))
  );
}

export function humanizeError(error: string | null): string | null {
  if (!error) return null;
  const first = namedLinks(error)[0];
  const shown = first ? `${first.link}${first.cut ? "…" : ""}` : null;
  const words = withoutLinks(error);
  for (const { match, hint, aboutLink, needsLink } of ERROR_HINTS) {
    const found = match.exec(words);
    if (found === null || (needsLink && shown === null)) continue;
    const sentence = hint(found);
    return aboutLink && shown !== null ? `${shown} — ${sentence}` : sentence;
  }
  return null;
}
