import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Report, RunDetail } from "../api/types";
import * as v2 from "../api/v2";
import RunView from "./RunView";

const doneDetail: RunDetail = {
  run: {
    id: "r",
    status: "DONE",
    created_at: "t",
    error: null,
    title: "Coastal Brief",
    source_count: 1,
    scores: null
  },
  job: null,
  uploads: []
};

const report: Report = {
  run_id: "r",
  title: "Coastal Brief",
  status: "DONE",
  report_doc_id: "reportdoc",
  scores: { accuracy: 1, coverage: 1, credibility: 0.8, validity: 0.6 },
  accuracy_detail: null,
  validity_detail: null,
  credibility_detail: null,
  stats: { claims_total: 2, claims_supported: 1, claims_contradicted: 0, claims_unverifiable: 1 },
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
    },
    {
      claim_id: "c2",
      text: "made up entirely",
      page: 2,
      value: null,
      unit: null,
      year: null,
      verdict: "UNVERIFIABLE",
      stance: "asserted",
      downgraded: false,
      quote: null,
      quote_verified: null,
      rationale: "no coverage",
      year_flag: null,
      evidence_source: null
    }
  ],
  sources: [{ doc_id: "sourcedoc", title: "Src", total: 80, tier: "VERIFIED_DOI", components: {}, metadata: {} }]
};

function renderAt(url: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[url]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunView />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  // jsdom has no object-URL support; the focus mode fetches PDFs as blobs.
  (URL as unknown as { createObjectURL: unknown }).createObjectURL = vi.fn(() => "blob:fake");
  (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = vi.fn();
});
afterEach(() => vi.restoreAllMocks());

describe("FocusClaims (via ?focus=claims)", () => {
  it("deep-links the report and source panes to the selected claim's pages", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue(doneDetail);
    vi.spyOn(v2, "getReport").mockResolvedValue(report);
    const blob = vi
      .spyOn(v2, "fetchPdfBlob")
      .mockResolvedValue(new Blob(["%PDF"], { type: "application/pdf" }));

    renderAt("/runs/r?focus=claims");
    await waitFor(() => expect(screen.getByTitle("report")).toBeInTheDocument());
    expect(screen.getByTitle("report")).toHaveAttribute(
      "src",
      "blob:fake#page=1&toolbar=0&navpanes=0&view=FitH"
    );
    await waitFor(() => expect(screen.getByTitle("source")).toBeInTheDocument());
    expect(screen.getByTitle("source")).toHaveAttribute(
      "src",
      "blob:fake#page=3&toolbar=0&navpanes=0&view=FitH"
    );
    expect(blob).toHaveBeenCalledWith("r", "reportdoc");
    expect(blob).toHaveBeenCalledWith("r", "sourcedoc");
    // The selected claim's rationale and quote sit in the strip below (the
    // fixture's claim text equals its quote, so both render "“hunger fell”").
    expect(screen.getByText("verbatim")).toBeInTheDocument();
    expect(screen.getAllByText("“hunger fell”").length).toBeGreaterThan(0);
  });

  it("honors verdict and claim params, and Close returns to the panels", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue(doneDetail);
    vi.spyOn(v2, "getReport").mockResolvedValue(report);
    vi.spyOn(v2, "fetchPdfBlob").mockResolvedValue(
      new Blob(["%PDF"], { type: "application/pdf" })
    );

    renderAt("/runs/r?focus=claims&verdict=UNVERIFIABLE&claim=c2");
    await waitFor(() => expect(screen.getByText("made up entirely")).toBeInTheDocument());
    // The supported claim is filtered out of the list.
    expect(screen.queryByText("hunger fell")).not.toBeInTheDocument();
    // No evidence pane for a claim without a quoted source.
    expect(screen.getByText("This claim has no quoted source evidence.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "✕ Close" }));
    await waitFor(() =>
      expect(screen.getByPlaceholderText("Ask about this verification…")).toBeInTheDocument()
    );
  });
});
