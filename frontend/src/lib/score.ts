// The one place the score color bands live: rings, credibility badges, and
// anything else painting a 0–100 score must agree on what "good" means.
export type ScoreBand = "hi" | "mid" | "lo";

export function scoreBand(pct: number): ScoreBand {
  return pct >= 70 ? "hi" : pct >= 40 ? "mid" : "lo";
}

export const BAND_COLORS: Record<ScoreBand, string> = {
  hi: "var(--color-success)",
  mid: "var(--color-warning)",
  lo: "var(--color-danger)"
};

// Why a scored run shows no credibility number: every source was one that
// can't be scored (images are listed, never averaged), or there were none.
export function credibilityGapHint(method: string | null | undefined): string {
  if (method === "no_scorable_sources") return "No scorable sources";
  if (method === "no_sources") return "No sources to score";
  return "Not scored";
}
