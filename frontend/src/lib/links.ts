// Links added as sources: the checks the upload dialog runs before anything is
// sent (the server remains the authority), and the helpers that turn a link
// into something readable.

export const MAX_LINK_LENGTH = 2048;

// youtube.com, youtube-nocookie.com, youtubekids.com and youtu.be — bare or
// under any subdomain (www., m., music., gaming., consent., ...); trailing
// root dots name the same host. Mirrors fetch.is_youtube_url exactly.
const YOUTUBE_DOMAINS = ["youtube.com", "youtube-nocookie.com", "youtubekids.com", "youtu.be"];

// Eleven characters, as a video id is, but YouTube's names for its playlist
// player (/embed/videoseries) and a channel's live page (/embed/live_stream).
export const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
const NOT_VIDEO_IDS = new Set(["videoseries", "live_stream"]);
// The paths whose one segment after the prefix is a video id, each with at
// most a trailing slash after the id.
const VIDEO_PATH = /^\/(?:shorts|live|embed|v|e)\/([A-Za-z0-9_-]{11})\/?$/;
const SHORT_LINK_PATH = /^\/([A-Za-z0-9_-]{11})\/?$/;

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

function stripTrailingDots(host: string): string {
  return host.replace(/\.+$/, "");
}

// True for a host that is one of YouTube's domains or any host under one
// (www., m., music., gaming., consent., ...). A link there is read as a
// single video or refused; it is never added as a web page.
function isYoutubeDomain(host: string): boolean {
  return YOUTUBE_DOMAINS.some((domain) => host === domain || host.endsWith(`.${domain}`));
}

function isYoutubeHost(hostname: string): boolean {
  return isYoutubeDomain(stripTrailingDots(hostname.toLowerCase()));
}

// The id of the one video a URL names, or null when its host isn't
// YouTube's, or it names no single video (a channel, a playlist, a search).
// Mirrors the server's fetch.youtube_video_id: the forms are watch?v= (one
// id, however often repeated), youtu.be/ID, and /shorts|live|embed|v|e/ID.
// `url.pathname` is already percent-encoded, so an escaped character in a
// path id fails the pattern instead of being decoded; `searchParams` decodes
// a query value the way the server's parse_qs does. The host is lowercased
// and its trailing dots stripped once, here, and reused for every check.
function videoIdOf(url: URL): string | null {
  const host = stripTrailingDots(url.hostname.toLowerCase());
  if (!isYoutubeDomain(host)) return null;
  let candidate: string | null;
  if (host === "youtu.be" || host.endsWith(".youtu.be")) {
    candidate = SHORT_LINK_PATH.exec(url.pathname)?.[1] ?? null;
  } else if (url.pathname === "/watch" || url.pathname === "/watch/") {
    const values = new Set(url.searchParams.getAll("v"));
    candidate = values.size === 1 ? [...values][0] : null;
  } else {
    candidate = VIDEO_PATH.exec(url.pathname)?.[1] ?? null;
  }
  if (candidate === null || !VIDEO_ID.test(candidate) || NOT_VIDEO_IDS.has(candidate)) {
    return null;
  }
  return candidate;
}

// A thin wrapper for callers with only a raw string in hand; `videoIdOf`
// above does the actual work from an already-parsed URL.
export function youtubeVideoId(raw: string): string | null {
  const url = parse(raw);
  return url === null ? null : videoIdOf(url);
}

// The one link a video is stored, compared and read by — every accepted
// spelling becomes this, so the same video twice is a duplicate however it
// was written.
export function canonicalVideoUrl(id: string): string {
  return `https://www.youtube.com/watch?v=${id}`;
}

// A link's comparable identity: its canonical video link when it names one,
// its canonical form otherwise, or an error when it is YouTube's but names
// no single video. `existing` links are always ones checkLink already
// accepted, so they resolve the same way — a duplicate is caught however
// either spelling was written.
function identity(url: URL): LinkCheck {
  if (isYoutubeHost(url.hostname)) {
    const id = videoIdOf(url);
    return id === null
      ? { error: "That YouTube link isn't a single video. Add the link of one video." }
      : { link: canonicalVideoUrl(id) };
  }
  return { link: canonical(url) };
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
  const found = identity(url);
  if ("error" in found) {
    return found;
  }
  const { link } = found;
  // Measured as sent, where one typed character can take several ("é" is
  // "%C3%A9"), so the refusal names no count the typed text might not reach.
  if (link.length > MAX_LINK_LENGTH) {
    return { error: "That link is too long to add." };
  }
  const isDuplicate = existing.some((added) => {
    const parsed = parse(added);
    if (parsed === null) return added === link;
    const addedFound = identity(parsed);
    return "link" in addedFound && addedFound.link === link;
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

// A decoded name is shown only when it can't be mistaken for another. Letters
// from scripts with Latin look-alikes — Cyrillic, Greek and others — can spell
// a whole name ("аррӏе" is all Cyrillic) or slip into a Latin one ("аpple"), and
// symbols can stand in for letters, so any of those keeps the encoded form.
// Allowed: Latin alone, or scripts with no Latin look-alikes without Latin.
const DISTINCT_SCRIPT_LABEL =
  /^[\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Hangul}\p{scx=Bopomofo}\p{scx=Arabic}\p{scx=Hebrew}\p{scx=Thai}\p{M}0-9-]+$/u;
// Latin is read letter by letter with its accents split off: ASCII letters, plus
// the few letters Latin languages add that can't pass for an ASCII one. Letters
// that read as plain ones (ı ȷ ɑ ɡ, ǀ, small capitals) and a dot above an i, j
// or l keep the encoded form.
const LATIN_LABEL = /^(?:[a-z0-9ßæøœłđ-][\u0300-\u036f]*)+$/u;
const DOTTED_STEM = /[ijl][\u0300-\u036f]*\u0307/u;
// A mark with no glyph (U+034F, the variation selectors) makes a familiar name
// out of another, in any script.
const INVISIBLE = /\p{Default_Ignorable_Code_Point}/u;
const ASCII_ONLY = /^[\x00-\x7f]*$/;

function readableLabel(label: string): string {
  if (!label.startsWith("xn--")) return label;
  const decoded = decodePunycode(label.slice(4));
  // A label that encodes nothing ("xn--apple-") reads as the name it isn't.
  // Registered names hold only code points compatibility normalization leaves
  // alone, so a label it changes (ſ, ａ, halfwidth katakana) is another spelling
  // of some plain name.
  if (
    decoded === null ||
    ASCII_ONLY.test(decoded) ||
    INVISIBLE.test(decoded) ||
    decoded.normalize("NFKC") !== decoded
  ) {
    return label;
  }
  const letters = decoded.normalize("NFD");
  const latin = LATIN_LABEL.test(letters) && !DOTTED_STEM.test(letters);
  return latin || DISTINCT_SCRIPT_LABEL.test(decoded) ? decoded : label;
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
// A YouTube video reads by its id instead ("youtube.com · dQw4w9WgXcQ") — its
// path is the same "/watch" for every video, so showing it would print one
// indistinguishable row per video.
export function linkHostPath(raw: string): string {
  const url = parse(raw);
  if (url === null) return raw;
  const id = videoIdOf(url);
  if (id !== null) return `${displayHost(url).replace(/^www\./, "")} · ${id}`;
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
