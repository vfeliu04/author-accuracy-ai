import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { Report } from "../api/types";
import FocusValidity from "./FocusValidity";

const report: Report = {
  run_id: "r",
  title: null,
  status: "DONE",
  report_doc_id: "d",
  scores: { accuracy: 1, coverage: 1, credibility: 0.8, validity: 0.42 },
  accuracy_detail: null,
  validity_detail: {
    components: {
      coverage: {
        score: 70,
        justification: "treats its stated scope",
        quote: "the report examines",
        quote_verified: 1
      },
      methodology: {
        score: 20,
        justification: "no methods section",
        quote: "figures show",
        quote_verified: 0
      },
      recency: { score: null, justification: "", quote: null, quote_verified: null }
    },
    weights_used: { coverage: 0.5, methodology: 0.5 }
  },
  credibility_detail: null,
  stats: { claims_total: 0, claims_supported: 0, claims_contradicted: 0, claims_unverifiable: 0 },
  claims: [],
  sources: []
};

describe("FocusValidity", () => {
  it("renders components with justification, quotes, and unverified-quote flags", () => {
    render(
      <MemoryRouter>
        <FocusValidity report={report} />
      </MemoryRouter>
    );
    expect(screen.getByText("coverage")).toBeInTheDocument();
    expect(screen.getByText("treats its stated scope")).toBeInTheDocument();
    expect(screen.getByText(/quote not found in the report/)).toBeInTheDocument();
    // Null recency explains the renormalization instead of showing a zero.
    expect(screen.getByText(/excluded/)).toBeInTheDocument();
  });

  it("explains when a run predates stored rubric details", () => {
    render(
      <MemoryRouter>
        <FocusValidity report={{ ...report, validity_detail: null }} />
      </MemoryRouter>
    );
    expect(screen.getByText(/scored before rubric details were stored/)).toBeInTheDocument();
  });
});
