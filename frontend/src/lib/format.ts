// Small pure formatters shared across the UI.
import type { EvidenceSource } from "../api/types";

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

// "1 link", "2 links".
export function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function pct(fraction: number | null | undefined): string {
  return fraction === null || fraction === undefined ? "—" : `${Math.round(fraction * 100)}%`;
}

// 12:34 under an hour, 1:02:03 beyond — how video players label time.
export function formatTimestamp(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = String(whole % 60).padStart(2, "0");
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}`
    : `${minutes}:${secs}`;
}

// Where in its source a quote sits, phrased by SOURCE TYPE and never inferred
// from which locator happens to be set: a PDF item can lack a page, and that
// must not read as a web section. Null when there is nothing to cite.
export function citeLabel(
  source: Pick<EvidenceSource, "source_type" | "page" | "section" | "start_seconds">
): string | null {
  switch (source.source_type) {
    case "web":
      return source.section ? `§ ${source.section}` : null;
    case "youtube":
      return source.start_seconds !== null ? formatTimestamp(source.start_seconds) : null;
    case "image":
      return "image";
    default:
      return source.page !== null ? `p.${source.page}` : null;
  }
}
