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
    // The server's own wording, not "TimeoutError" anywhere before "Batch": that
    // rescans the rest of the message from every "TimeoutError" a site repeats.
    match: /TimeoutError: Batch\b|still 'in_progress'/i,
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
    match: /too large or complex/,
    aboutLink: true,
    needsLink: true,
    hint: () => "That page is too large or complex to read in time."
  },
  // Any other way reading an arrived page fails: the reader stopped, or raised
  // something unexpected. Its inner text must not pick a hint below.
  {
    match: /could not be read: /,
    aboutLink: true,
    needsLink: true,
    hint: () =>
      "Reading that page failed unexpectedly. Retrying may help; if it keeps failing, remove that link."
  },
  // The ways a link fails before its page arrives, each keyed to the server's
  // own wording. They come before the generic status and timeout phrasing: a
  // redirect with no address also names its HTTP status.
  {
    match: /could not be resolved|resolved to no addresses|returned an unparseable address/,
    aboutLink: true,
    needsLink: true,
    hint: () =>
      "That site couldn't be found. Check the link for typos, or the internet connection."
  },
  // A certificate failure is also a ConnectError, so it is recognized first.
  {
    match: /CERTIFICATE_VERIFY_FAILED|certificate verify failed|\[SSL[:\]]|_ssl\.c/i,
    aboutLink: true,
    needsLink: true,
    hint: () => "A secure connection to the site couldn't be established."
  },
  {
    match: /failed: \w*(?:Connect|Read|Write|Protocol|Network|Close)Error\b/,
    aboutLink: true,
    needsLink: true,
    hint: () => "The site couldn't be reached, or it dropped the connection."
  },
  {
    match: /exceeds the [\d,]+-byte limit/,
    aboutLink: true,
    needsLink: true,
    hint: () => "That page is too large to read."
  },
  {
    match: /more than \d+ redirects/,
    aboutLink: true,
    needsLink: true,
    hint: () => "That link redirects too many times to follow."
  },
  {
    match:
      /refused redirect|invalid redirect Location|redirected to an invalid URL|redirect without a Location/,
    aboutLink: true,
    needsLink: true,
    hint: () => "That link redirects to an address that can't be opened."
  },
  {
    match: /body is not a PDF/,
    aboutLink: true,
    needsLink: true,
    hint: () => "The site says that link is a PDF, but the file it sent isn't one."
  },
  {
    match: /unsupported (?:stacked )?Content-Encoding|\bDecodingError\b|malformed Content-Length/,
    aboutLink: true,
    needsLink: true,
    hint: () => "The site sent that page in a form that can't be read."
  },
  {
    match: /is not valid UTF-8|declares an unknown charset/,
    aboutLink: true,
    needsLink: true,
    hint: () => "That page's text is in an encoding that can't be read."
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
// mark come off the end until the last character belongs to the link. The
// counts are taken once and kept current as characters come off, so a long
// run of brackets a site sent costs one pass, not one pass per bracket.
function trimLink(raw: string): NamedLink {
  let end = raw.length;
  let cut = false;
  let quotes = count(raw, "'");
  let unopenedParens = count(raw, ")") - count(raw, "(");
  let unopenedBrackets = count(raw, "]") - count(raw, "[");
  for (;;) {
    const last = raw.charAt(end - 1);
    if (end >= CUT_MARK.length && raw.startsWith(CUT_MARK, end - CUT_MARK.length)) {
      end -= CUT_MARK.length;
      cut = true;
    } else if (/[.,;:!?]/.test(last)) {
      end -= 1;
    } else if (last === "'" && quotes % 2 === 1) {
      quotes -= 1;
      end -= 1;
    } else if (last === ")" && unopenedParens > 0) {
      unopenedParens -= 1;
      end -= 1;
    } else if (last === "]" && unopenedBrackets > 0) {
      unopenedBrackets -= 1;
      end -= 1;
    } else {
      return { link: raw.slice(0, end), cut };
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

// A link's host as the message spells it: between the scheme and the path,
// without a port or an IPv6 literal's brackets.
function hostOf(link: string): string {
  const authority = link.replace(/^https?:\/\//i, "").split(/[/?#]/, 1)[0];
  if (authority.startsWith("[")) return authority.slice(1, authority.indexOf("]"));
  return authority.split(":", 1)[0];
}

// The message with every link blanked out, so words in an address
// (".../billing", ".../batch-jobs", ".../deadlines") never pick a translation.
// The server also repeats a link's host on its own, in quotes
// ("'billing.example.org' could not be resolved"), so that copy goes too.
// `named` is namedLinks(error): replace visits the same matches in the same order.
function withoutLinks(error: string, named: readonly NamedLink[]): string {
  let index = 0;
  let words = error.replace(LINK, (raw) => ` ${raw.slice(named[index++].link.length)}`);
  for (const { link } of named) {
    const host = hostOf(link);
    if (host !== "") words = words.split(`'${host}'`).join(" ");
  }
  return words;
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
  const named = namedLinks(error);
  const first = named[0];
  const shown = first ? `${first.link}${first.cut ? "…" : ""}` : null;
  const words = withoutLinks(error, named);
  for (const { match, hint, aboutLink, needsLink } of ERROR_HINTS) {
    const found = match.exec(words);
    if (found === null || (needsLink && shown === null)) continue;
    const sentence = hint(found);
    return aboutLink && shown !== null ? `${shown} — ${sentence}` : sentence;
  }
  return null;
}
