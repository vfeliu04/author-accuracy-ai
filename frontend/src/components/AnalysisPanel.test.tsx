import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Report } from "../api/types";
import AnalysisPanel from "./AnalysisPanel";

const baseReport: Report = {
  run_id: "r",
  title: null,
  status: "DONE",
  report_doc_id: "d",
  scores: { accuracy: 1, coverage: 0.5, credibility: 0.8, validity: null },
  accuracy_detail: null,
  validity_detail: null,
  credibility_detail: null,
  stats: { claims_total: 5, claims_supported: 3, claims_contradicted: 1, claims_unverifiable: 1 },
  claims: [],
  sources: []
};

describe("AnalysisPanel", () => {
  it("renders the four rings with values and em-dash for null metrics", () => {
    render(<AnalysisPanel report={baseReport} onOpenClaims={vi.fn()} />);
    // Ring labels ("Credibility"/"Validity" also appear as tile names).
    for (const label of ["Accuracy", "Coverage", "Credibility", "Validity"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    expect(screen.getByText("100")).toBeInTheDocument(); // accuracy 1 → 100
    expect(screen.getByText("—")).toBeInTheDocument(); // validity null
    // No composite score — the three metrics stay independent by design.
    expect(screen.queryByText(/overall/i)).not.toBeInTheDocument();
  });

  it("shows ghost state before the run is scored", () => {
    render(<AnalysisPanel report={undefined} />);
    expect(
      screen.getByText("Scores, claims and chat appear here when verification completes.")
    ).toBeInTheDocument();
  });
});
