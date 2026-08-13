import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RatingPanel from "./RatingPanel";

describe("RatingPanel", () => {
  it("renders four metric rings from fractions and no composite overall", () => {
    render(<RatingPanel scores={{ accuracy: 1, coverage: 0.5, credibility: 0.8, validity: 0.6 }} />);
    for (const label of ["Accuracy", "Coverage", "Credibility", "Validity"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    // v2 has no composite score — the "Overall" bar must be gone.
    expect(screen.queryByText("Overall")).not.toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument(); // accuracy 1.0 → 100
  });

  it("shows a scoring state when scores are null", () => {
    render(<RatingPanel scores={null} />);
    expect(screen.getByText(/Scoring in progress/)).toBeInTheDocument();
  });

  it("renders a dash for a null metric (nothing decided)", () => {
    render(<RatingPanel scores={{ accuracy: null, coverage: 0.5, credibility: 0.8, validity: 0.6 }} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
