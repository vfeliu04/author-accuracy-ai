// Raw exception text is for logs; users get the translation (the raw message
// stays visible in a collapsible block wherever this is rendered).

import { youtubeVideoId } from "./links";

type ErrorHint = {
  match: RegExp;
  // `about.video`: the link the message names is a YouTube video.
  hint: (found: RegExpExecArray, about: { video: boolean }) => string;
  // "prefix": a link failure names the link, so the reader knows which one to
  // fix. "required": the same, and the hint is skipped when the message names
  // no link — "HTTP 404" and "timed out" are generic network phrasing that
  // describe a link only when the message names one.
  link?: "prefix" | "required";
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
    link: "prefix",
    hint: () => "That link points to a private network address, so it can't be opened."
  },
  {
    match: /unsupported content type/i,
    link: "prefix",
    hint: () => "That link isn't a web page or PDF."
  },
  {
    match: /no readable article text/i,
    link: "prefix",
    hint: () =>
      "That page has no readable text. Pages that need JavaScript to show their content can't be read."
  },
  {
    match: /too large or complex/,
    link: "required",
    hint: () => "That page is too large or complex to read in time."
  },
  // A YouTube video source (video.py): its message always names the video's
  // own canonical link via _cannot/_could_not, so these come before the
  // generic "could not be read: " catch-all below, which would otherwise
  // swallow the deno-runtime failure with a "page" message that fits neither.
  {
    match: /cannot be read: it has no usable captions/,
    link: "required",
    hint: () =>
      "That video has no usable captions — neither ones written for it nor YouTube's automatic captions in its spoken language — so it can't be checked."
  },
  {
    match: /YouTube asked to confirm the reader is not a bot/,
    link: "required",
    hint: () =>
      "YouTube asked to confirm the reader isn't a bot (it does this for live streams, and when it limits requests from this network). Try again later."
  },
  {
    match: /it is live now|it has not started yet|it has just ended/,
    link: "required",
    hint: () =>
      "That video isn't finished yet (it's live, hasn't started, or just ended), so YouTube hasn't written its captions. Try again later."
  },
  {
    match: /YouTube withheld this video's captions/,
    link: "required",
    hint: () => "YouTube withheld this video's captions from the reader. Try again later."
  },
  {
    match: /cannot be read: (?:Private video|Video unavailable)/,
    link: "required",
    hint: () => "YouTube says that video is private or has been removed, so it can't be read."
  },
  {
    match: /took longer than [\d.]+ seconds to read — YouTube may be slow to answer/,
    link: "required",
    hint: () => "Reading that video took too long — YouTube may be slow to answer. Try again later."
  },
  {
    match: /its captions are larger than [\d,]+ bytes/,
    link: "required",
    hint: () => "That video's captions are larger than the reader can handle."
  },
  {
    match: /could not use its JavaScript runtime \(deno\)/,
    link: "required",
    hint: () =>
      "The verification server's video reader couldn't start its JavaScript runtime. This is a server problem, not the link — let the operator know."
  },
  {
    // The server names the video it was about to read, like its "could not
    // use" neighbor above.
    match: /JavaScript runtime \(deno\) is not installed/,
    link: "required",
    hint: () =>
      "The verification server's video reader isn't fully installed. This is a server problem, not the link — let the operator know."
  },
  {
    match: /redirects to the YouTube video/,
    link: "required",
    hint: () =>
      "That link redirects to a YouTube video — add the video's own link as a source instead of the web page."
  },
  {
    match: /redirects to a YouTube page/,
    link: "required",
    hint: () =>
      "That link redirects to a YouTube page (a channel, playlist or search page), which can't be read as a source."
  },
  {
    // YouTube rate-limiting the video reader: yt-dlp's own words, inside the
    // reader's failure, which the catch-all below would otherwise take.
    match: /The YouTube video\s+could not be read: [^]*?\bHTTP (?:Error )?429\b/,
    link: "required",
    hint: () => "YouTube is limiting requests from this network. Try again later."
  },
  // Any other way reading an arrived page or a video fails: the reader
  // stopped, or raised something unexpected. Its inner text must not pick a
  // hint below.
  {
    match: /could not be read: /,
    link: "required",
    hint: (_found, { video }) =>
      video
        ? "Reading that video failed unexpectedly. Retrying may help; if it keeps failing, remove that video."
        : "Reading that page failed unexpectedly. Retrying may help; if it keeps failing, remove that link."
  },
  // The ways a link fails before its page arrives, each keyed to the server's
  // own wording. They come before the generic status and timeout phrasing: a
  // redirect with no address also names its HTTP status.
  {
    match: /could not be resolved|resolved to no addresses|returned an unparseable address/,
    link: "required",
    hint: () =>
      "That site couldn't be found. Check the link for typos, or the internet connection."
  },
  // A certificate failure is also a ConnectError, so it is recognized first.
  {
    match: /CERTIFICATE_VERIFY_FAILED|certificate verify failed|\[SSL[:\]]|_ssl\.c/i,
    link: "required",
    hint: () => "A secure connection to the site couldn't be established."
  },
  {
    match: /failed: \w*(?:Connect|Read|Write|Protocol|Network|Close)Error\b/,
    link: "required",
    hint: () => "The site couldn't be reached, or it dropped the connection."
  },
  {
    match: /exceeds the [\d,]+-byte limit/,
    link: "required",
    hint: () => "That page is too large to read."
  },
  {
    match: /more than \d+ redirects/,
    link: "required",
    hint: () => "That link redirects too many times to follow."
  },
  {
    match:
      /refused redirect|invalid redirect Location|redirected to an invalid URL|redirect without a Location/,
    link: "required",
    hint: () => "That link redirects to an address that can't be opened."
  },
  {
    match: /body is not a PDF/,
    link: "required",
    hint: () => "The site says that link is a PDF, but the file it sent isn't one."
  },
  {
    match: /unsupported (?:stacked )?Content-Encoding|\bDecodingError\b|malformed Content-Length/,
    link: "required",
    hint: () => "The site sent that page in a form that can't be read."
  },
  {
    match: /is not valid UTF-8|declares an unknown charset/,
    link: "required",
    hint: () => "That page's text is in an encoding that can't be read."
  },
  {
    match: /\bHTTP (\d{3})\b/,
    link: "required",
    hint: (found) => `The site returned an error (${found[1]}).`
  },
  {
    match: /timed out|deadline/i,
    link: "required",
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

// The links a message names (as namedLinks reads them), and the message with
// every link blanked out, so words in an address (".../billing",
// ".../batch-jobs", ".../deadlines") never pick a translation. The server also
// repeats a link's host on its own, in quotes ("'billing.example.org' could
// not be resolved"), so that copy goes too. One pass: each link is trimmed
// once, where it is blanked.
function scanLinks(error: string): { named: NamedLink[]; words: string } {
  const named: NamedLink[] = [];
  let words = error.replace(LINK, (raw) => {
    const found = trimLink(raw);
    named.push(found);
    return ` ${raw.slice(found.link.length)}`;
  });
  for (const { link } of named) {
    const host = hostOf(link);
    if (host !== "") words = words.split(`'${host}'`).join(" ");
  }
  return { named, words };
}

function appearsWhole(error: string, link: string, links: readonly string[]): boolean {
  if (link === "") return false;
  for (let at = error.indexOf(link); at !== -1; at = error.indexOf(link, at + 1)) {
    const longer = links.some((other) => other.length > link.length && error.startsWith(other, at));
    if (!longer) return true;
  }
  return false;
}

// A link as the server quotes it: Python's repr doubles each backslash, which a
// query string may keep ("?q=C:\Users").
function asQuoted(link: string): string {
  return link.split("\\").join("\\\\");
}

// Which of a run's added links a failure message names. A link counts where
// it appears whole, unless a longer added link starting the same way is what
// appears there (".../report" inside ".../report-2024"), or where the message
// cut a long link short and the added link starts with what is left. Each link
// is looked for as added and as quoted.
export function linksNamedIn(error: string, links: readonly string[]): string[] {
  // A page that redirects into YouTube is refused naming the video it leads
  // to, as the link to add instead: that video is not what failed.
  error = error.replace(/redirects to the YouTube video \S+/g, "redirects to the YouTube video");
  const cutStarts = namedLinks(error)
    .filter((named) => named.cut)
    .map((named) => named.link);
  const quoted = links.map(asQuoted);
  const spellings = [...links, ...quoted];
  return links.filter((link, index) =>
    (quoted[index] === link ? [link] : [link, quoted[index]]).some(
      (spelled) =>
        appearsWhole(error, spelled, spellings) ||
        cutStarts.some((start) => spelled.startsWith(start))
    )
  );
}

export function humanizeError(error: string | null): string | null {
  if (!error) return null;
  const { named, words } = scanLinks(error);
  const first = named[0];
  const shown = first ? `${first.link}${first.cut ? "…" : ""}` : null;
  const video = first !== undefined && youtubeVideoId(first.link) !== null;
  for (const { match, hint, link } of ERROR_HINTS) {
    const found = match.exec(words);
    if (found === null || (link === "required" && shown === null)) continue;
    const sentence = hint(found, { video });
    return link !== undefined && shown !== null ? `${shown} — ${sentence}` : sentence;
  }
  return null;
}
