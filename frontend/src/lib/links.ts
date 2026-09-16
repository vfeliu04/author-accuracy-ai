// Links added as sources: the checks the upload dialog runs before anything is
// sent (the server remains the authority), and the helpers that turn a link
// into something readable.

export const MAX_LINK_LENGTH = 2048;

// youtube.com, youtu.be and youtube-nocookie.com — bare or on the www., m. and
// music. subdomains; any trailing root dots name the same host.
const YOUTUBE_HOST = /^(?:(?:www|m|music)\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)\.*$/;

// The characters the server allows in a site name. A parsed link spells an
// international name in its encoded ASCII form, so this holds for those too.
const HOST_CHARS = /^[a-z0-9._-]+$/i;

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
  if (url.username || url.password) {
    return { error: "Links with a username or password can't be added." };
  }
  if (url.port === "0") {
    return { error: "That link has an invalid port." };
  }
  // An IPv6 address is bracketed, and the parser has already checked it.
  if (!url.hostname.startsWith("[") && !HOST_CHARS.test(url.hostname)) {
    return { error: "That link's site name isn't valid." };
  }
  if (YOUTUBE_HOST.test(url.hostname)) {
    return { error: "YouTube links aren't supported yet." };
  }
  // Measured as sent, where one typed character can take several ("é" is
  // "%C3%A9"), so the refusal names no count the typed text might not reach.
  const link = canonical(url);
  if (link.length > MAX_LINK_LENGTH) {
    return { error: "That link is too long to add." };
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

// Punycode (RFC 3492), decoding only: a parsed link spells an international
// host ("bücher.de") as "xn--bcher-kva.de", and people should read the former.
const BASE = 36;
const T_MIN = 1;
const T_MAX = 26;

function adapt(delta: number, points: number, first: boolean): number {
  let scaled = Math.floor(delta / (first ? 700 : 2));
  scaled += Math.floor(scaled / points);
  let k = 0;
  while (scaled > ((BASE - T_MIN) * T_MAX) / 2) {
    scaled = Math.floor(scaled / (BASE - T_MIN));
    k += BASE;
  }
  return k + Math.floor(((BASE - T_MIN + 1) * scaled) / (scaled + 38));
}

function digitOf(code: number): number {
  if (code >= 48 && code <= 57) return code - 22; // 0-9 are 26-35
  if (code >= 65 && code <= 90) return code - 65;
  if (code >= 97 && code <= 122) return code - 97;
  return BASE;
}

function decodePunycode(encoded: string): string | null {
  const delimiter = encoded.lastIndexOf("-");
  const output = Array.from(delimiter > 0 ? encoded.slice(0, delimiter) : "", (char) =>
    char.charCodeAt(0)
  );
  if (output.some((code) => code >= 0x80)) return null;
  let n = 0x80;
  let bias = 72;
  let i = 0;
  let pos = delimiter > 0 ? delimiter + 1 : 0;
  while (pos < encoded.length) {
    const before = i;
    for (let w = 1, k = BASE; ; k += BASE) {
      if (pos >= encoded.length) return null;
      const digit = digitOf(encoded.charCodeAt(pos++));
      if (digit >= BASE) return null;
      i += digit * w;
      const t = k <= bias ? T_MIN : k >= bias + T_MAX ? T_MAX : k - bias;
      if (digit < t) break;
      w *= BASE - t;
    }
    const points = output.length + 1;
    bias = adapt(i - before, points, before === 0);
    n += Math.floor(i / points);
    i %= points;
    if (n > 0x10ffff) return null;
    output.splice(i, 0, n);
    i += 1;
  }
  return String.fromCodePoint(...output);
}

// A decoded name is shown only when it can't pass for a familiar one. Letters
// from scripts with Latin look-alikes — Cyrillic, Greek and others — can spell
// a whole name ("аррӏе" is all Cyrillic) or slip into a Latin one ("аpple"), and
// symbols can stand in for letters, so any of those keeps the encoded form.
// Allowed: Latin alone, or scripts with no Latin look-alikes without Latin.
const LATIN_LABEL = /^[\p{Script=Latin}\p{M}0-9-]+$/u;
const DISTINCT_SCRIPT_LABEL =
  /^[\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Hangul}\p{scx=Bopomofo}\p{scx=Arabic}\p{scx=Hebrew}\p{scx=Thai}\p{M}0-9-]+$/u;

function readableLabel(label: string): string {
  if (!label.startsWith("xn--")) return label;
  const decoded = decodePunycode(label.slice(4));
  const safe =
    decoded !== null && (LATIN_LABEL.test(decoded) || DISTINCT_SCRIPT_LABEL.test(decoded));
  return safe ? decoded : label;
}

// The host as people write it, port included.
function displayHost(url: URL): string {
  const name = url.hostname.split(".").map(readableLabel).join(".");
  return url.port ? `${name}:${url.port}` : name;
}

export function linkHost(raw: string): string {
  const url = parse(raw);
  return url === null ? raw : displayHost(url);
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
  const host = displayHost(url);
  return path === "/" ? host : `${host}${path}`;
}

// A source's display name: its title, else its link as host and path, else
// the caller's fallback. A link page with no title of its own is stored under
// the link itself, which reads as no title.
export function sourceName(
  source: { title: string | null; url: string | null },
  fallback: string
): string {
  const title = source.title && source.title !== source.url ? source.title : null;
  return title || (source.url ? linkHostPath(source.url) : fallback);
}

// The link as an href only when it is http(s) — anything else (javascript:,
// data:, a bare path) must never become clickable.
export function safeHttpUrl(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const url = parse(raw);
  return url !== null && (url.protocol === "http:" || url.protocol === "https:") ? url.href : null;
}
