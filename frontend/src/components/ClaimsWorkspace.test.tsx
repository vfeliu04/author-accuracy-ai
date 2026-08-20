import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Report } from "../api/types";
import * as v2 from "../api/v2";

const report: Report = {
  run_id: "r",
  status: "DONE",
  report_doc_id: "reportdoc",
  scores: { accuracy: 1, coverage: 1, credibility: 0.8, validity: 0.6 },
  stats: { claims_total: 1, claims_supported: 1, claims_contradicted: 0, claims_unverifiable: 0 },
  claims: [
    {
      claim_id: "c1",
      text: "hunger fell",
      page: 1,
      value: null,
      unit: null,
      year: null,
      verdict: "SUPPORTED",
      stance: "asserted",
      downgraded: false,
      quote: "hunger fell",
      quote_verified: 1,
      rationale: "verbatim",
      year_flag: null,
      evidence_source: { doc_id: "sourcedoc", title: "Src", page: 3 }
    }
  ],
  sources: [{ doc_id: "sourcedoc", title: "Src", total: 80, tier: "VERIFIED_DOI", components: {} }]
};

beforeEach(() => {
  // jsdom has no object-URL support; the workspace fetches PDFs as blobs.
  (URL as unknown as { createObjectURL: unknown }).createObjectURL = vi.fn(() => "blob:fake");
  (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = vi.fn();
});
afterEach(() => vi.restoreAllMocks());

describe("ClaimsWorkspace", () => {
  it("deep-links the report and source panes to the claim's pages", async () => {
    vi.spyOn(v2, "getReport").mockResolvedValue(report);
    vi.spyOn(v2, "fetchPdfBlob").mockResolvedValue(
      new Blob([new Uint8Array([1])], { type: "application/pdf" })
    );
    const { default: ClaimsWorkspace } = await import("./ClaimsWorkspace");

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={["/runs/r/workspace"]}>
          <Routes>
            <Route path="/runs/:runId/workspace" element={<ClaimsWorkspace />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );

    // The selected claim's text appears twice by design: once in the list row
    // and once in the detail strip that surfaces rationale + quote.
    await waitFor(() => expect(screen.getAllByText("hunger fell").length).toBeGreaterThan(0));
    expect(screen.getByText("verbatim")).toBeInTheDocument(); // rationale is on screen now
    // Report pane deep-links to the claim's page; source pane to the evidence page.
    await waitFor(() =>
      expect(screen.getByTitle("report")).toHaveAttribute("src", "blob:fake#page=1")
    );
    expect(screen.getByTitle("source")).toHaveAttribute("src", "blob:fake#page=3");
    // Both panes requested exactly their own document.
    expect(v2.fetchPdfBlob).toHaveBeenCalledWith("r", "reportdoc");
    expect(v2.fetchPdfBlob).toHaveBeenCalledWith("r", "sourcedoc");
  });
});
