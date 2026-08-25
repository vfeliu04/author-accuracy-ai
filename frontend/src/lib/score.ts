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
