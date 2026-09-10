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

// The first link a message names, without the quotes or sentence punctuation
// wrapped around it.
export function namedLink(error: string): string | null {
  const found = /https?:\/\/[^\s'"<>()[\]]+/i.exec(error);
  return found ? found[0].replace(/[.,;:!?]+$/, "") : null;
}

export function humanizeError(error: string | null): string | null {
  if (!error) return null;
  const link = namedLink(error);
  for (const { match, hint, aboutLink, needsLink } of ERROR_HINTS) {
    const found = match.exec(error);
    if (found === null || (needsLink && link === null)) continue;
    const sentence = hint(found);
    return aboutLink && link !== null ? `${link} — ${sentence}` : sentence;
  }
  return null;
}
