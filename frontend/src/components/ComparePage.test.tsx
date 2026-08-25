import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Report } from "../api/types";
import * as v2 from "../api/v2";
import ComparePage from "./ComparePage";

function reportFor(id: string, overrides: Partial<Report>): Report {
  return {
    run_id: id,
    title: null,
    status: "DONE",
    report_doc_id: "d",
    scores: { accuracy: 0.6, coverage: 0.5, credibility: 0.8, validity: 0.6 },
    accuracy_detail: null,
    validity_detail: null,
    credibility_detail: null,
    stats: { claims_total: 10, claims_supported: 6, claims_contradicted: 2, claims_unverifiable: 2 },
    claims: [],
    sources: [],
    ...overrides
  };
}

function renderCompare(a: string, b: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/compare?a=${a}&b=${b}`]}>
        <Routes>
          <Route path="/compare" element={<ComparePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("ComparePage", () => {
  it("shows per-metric deltas between two runs", async () => {
    vi.spyOn(v2, "getReport").mockImplementation((id: string) =>
      Promise.resolve(
        id === "a"
          ? reportFor("a", {})
          : reportFor("b", {
              scores: { accuracy: 0.8, coverage: 0.5, credibility: 0.6, validity: 0.6 },
              stats: { claims_total: 12, claims_supported: 9, claims_contradicted: 1, claims_unverifiable: 2 }
            })
      )
    );
    renderCompare("a", "b");
    // accuracy 0.6 → 0.8 = +20%; credibility 0.8 → 0.6 = -20%;
    // supported 6 → 9 = +3; contradicted 2 → 1 = -1.
    await waitFor(() => expect(screen.getByText("+20%")).toBeInTheDocument());
    expect(screen.getByText("-20%")).toBeInTheDocument();
    expect(screen.getByText("+3")).toBeInTheDocument();
    expect(screen.getByText("-1")).toBeInTheDocument();
    // Scores keep valence colors; counts must stay neutral — fewer
    // unverifiable claims is not "bad", so no red/green on count deltas.
    expect(screen.getByText("+20%").className).toContain("compare__delta--up");
    expect(screen.getByText("+3").className).toContain("compare__delta--flat");
    expect(screen.getByText("-1").className).toContain("compare__delta--flat");
  });

  it("keeps the delta consistent with the two displayed cells", async () => {
    // 0.144 → 0.156 renders as 14% → 16% (a 2-point cell gap); the delta must
    // read the difference of the CELLS (+2%), not round(0.156-0.144)=+1%.
    vi.spyOn(v2, "getReport").mockImplementation((id: string) =>
      Promise.resolve(
        id === "a"
          ? reportFor("a", { scores: { accuracy: 0.144, coverage: 0.5, credibility: 0.8, validity: 0.6 } })
          : reportFor("b", { scores: { accuracy: 0.156, coverage: 0.5, credibility: 0.8, validity: 0.6 } })
      )
    );
    renderCompare("a", "b");
    await waitFor(() => expect(screen.getByText("14%")).toBeInTheDocument());
    expect(screen.getByText("16%")).toBeInTheDocument();
    expect(screen.getByText("+2%")).toBeInTheDocument();
  });

  it("guards an unscored run with dashes and a note", async () => {
    vi.spyOn(v2, "getReport").mockImplementation((id: string) =>
      Promise.resolve(id === "a" ? reportFor("a", {}) : reportFor("b", { status: "RUNNING", scores: null }))
    );
    renderCompare("a", "b");
    await waitFor(() => expect(screen.getByText(/not scored yet/)).toBeInTheDocument());
  });

  it("prompts to pick two runs when params are missing", () => {
    renderCompare("", "");
    expect(screen.getByText(/Pick two runs/)).toBeInTheDocument();
  });
});
