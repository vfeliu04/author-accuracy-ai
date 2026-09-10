// Links added as sources: the checks the upload dialog runs before anything is
// sent (the server remains the authority), and the helpers that turn a link
// into something readable.

export const MAX_LINK_LENGTH = 2048;

// youtube.com, youtu.be and youtube-nocookie.com — bare or on the www., m. and
// music. subdomains; a trailing root dot names the same host.
const YOUTUBE_HOST = /^(?:(?:www|m|music)\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)\.?$/;

export type LinkCheck = { link: string } | { error: string };

function parse(raw: string): URL | null {
  try {
    return new URL(raw);
  } catch {
    return null;
  }
}

// The one spelling links are compared and sent in: parsed, so host case or a
// bare origin's missing slash can't make one page look like two, and without
// the #fragment, which never reaches the site anyway.
function canonical(url: URL): string {
  const copy = new URL(url.href);
  copy.hash = "";
  return copy.href;
}

export function checkLink(input: string, existing: readonly string[]): LinkCheck {
  const url = parse(input.trim());
  if (url === null) {
    return { error: "That isn't a valid link. Paste the full address, starting with https://." };
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { error: "Only http:// and https:// links can be added." };
  }
  if (YOUTUBE_HOST.test(url.hostname)) {
    return { error: "YouTube links aren't supported yet." };
  }
  const link = canonical(url);
  if (link.length > MAX_LINK_LENGTH) {
    return { error: "That link is too long. Links can be at most 2,048 characters." };
  }
  const isDuplicate = existing.some((added) => {
    const parsed = parse(added);
    return (parsed ? canonical(parsed) : added) === link;
  });
  if (isDuplicate) {
    return { error: "That link is already added." };
  }
  return { link };
}

export function linkHost(raw: string): string {
  return parse(raw)?.host ?? raw;
}

// "example.org/water/report 2024": the host plus a decoded path, for reading.
export function linkHostPath(raw: string): string {
  const url = parse(raw);
  if (url === null) return raw;
  let path = url.pathname;
  try {
    path = decodeURIComponent(path);
  } catch {
    // Malformed escapes stay exactly as the link spells them.
  }
  return path === "/" ? url.host : `${url.host}${path}`;
}

// A source's display name: its title, else its link as host and path, else
// the caller's fallback.
export function sourceName(
  source: { title: string | null; url: string | null },
  fallback: string
): string {
  return source.title || (source.url ? linkHostPath(source.url) : fallback);
}

// The link as an href only when it is http(s) — anything else (javascript:,
// data:, a bare path) must never become clickable.
export function safeHttpUrl(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const url = parse(raw);
  return url !== null && (url.protocol === "http:" || url.protocol === "https:") ? url.href : null;
}
