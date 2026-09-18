import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Claim, PageSnapshot, Report, RunDetail } from "../api/types";
import * as v2 from "../api/v2";
import RunView from "./RunView";

const doneDetail: RunDetail = {
  run: {
    id: "r",
    status: "DONE",
    created_at: "t",
    error: null,
    title: "Coastal Brief",
    source_count: 3,
    scores: null
  },
  job: null,
  uploads: []
};

function claim(overrides: Partial<Claim> & Pick<Claim, "claim_id" | "text">): Claim {
  return {
    page: 1,
    value: null,
    unit: null,
    year: null,
    verdict: "SUPPORTED",
    stance: "asserted",
    downgraded: false,
    quote: null,
    quote_verified: null,
    rationale: "",
    year_flag: null,
    evidence_source: null,
    ...overrides
  };
}

const report: Report = {
  run_id: "r",
  title: "Coastal Brief",
  status: "DONE",
  report_doc_id: "reportdoc",
  scores: { accuracy: 1, coverage: 1, credibility: 0.8, validity: 0.6 },
  accuracy_detail: null,
  validity_detail: null,
  credibility_detail: null,
  stats: { claims_total: 4, claims_supported: 3, claims_contradicted: 0, claims_unverifiable: 1 },
  claims: [
    claim({
      claim_id: "c1",
      text: "hunger fell",
      quote: "hunger fell",
      quote_verified: 1,
      rationale: "verbatim",
      evidence_source: {
        doc_id: "sourcedoc",
        title: "Src",
        page: 3,
        source_type: "pdf",
        url: null,
        section: null,
        start_seconds: null,
        chunk_id: 11
      }
    }),
    claim({
      claim_id: "c2",
      text: "made up entirely",
      page: 2,
      verdict: "UNVERIFIABLE",
      rationale: "no coverage"
    }),
    claim({
      claim_id: "c3",
      text: "two billion people lack safe water",
      page: 2,
      quote: "Two billion people lack safe water",
      quote_verified: 1,
      rationale: "the page states it",
      evidence_source: {
        doc_id: "webdoc",
        title: "Water in Crisis",
        page: null,
        source_type: "web",
        url: "https://example.org/water",
        section: "Findings",
        start_seconds: null,
        chunk_id: 12
      }
    }),
    claim({
      claim_id: "c4",
      text: "the lecture confirms it",
      page: 3,
      quote: "it is confirmed",
      quote_verified: 1,
      rationale: "said in the talk",
      evidence_source: {
        doc_id: "videodoc",
        title: null,
        page: null,
        source_type: "youtube",
        url: "https://www.youtube.com/watch?v=abc123def45",
        section: null,
        start_seconds: 3723,
        chunk_id: 13
      }
    })
  ],
  sources: [
    {
      doc_id: "sourcedoc",
      title: "Src",
      source_type: "pdf",
      url: null,
      scorable: true,
      total: 80,
      tier: "VERIFIED_DOI",
      components: {},
      metadata: {},
      truncated: null
    }
  ]
};

const page: PageSnapshot = {
  schema: 1,
  document: {
    title: "Water in Crisis",
    sections: [
      { title: "Findings", page: null, text: "Two billion people lack safe water at home." }
    ]
  },
  provenance: {
    url: "https://example.org/water",
    final_url: "https://example.org/water",
    fetched_at: "2026-09-01T10:00:00Z",
    content_type: "text/html",
    title: "Water in Crisis",
    authors: [],
    publisher: null,
    publication_date: null,
    doi: null,
    scholarly: false
  }
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

function mockRun() {
  vi.spyOn(v2, "getRun").mockResolvedValue(doneDetail);
  vi.spyOn(v2, "getReport").mockResolvedValue(report);
  return vi
    .spyOn(v2, "fetchPdfBlob")
    .mockResolvedValue(new Blob(["%PDF"], { type: "application/pdf" }));
}

beforeEach(() => {
  // jsdom has no object-URL support; the focus mode fetches PDFs as blobs.
  (URL as unknown as { createObjectURL: unknown }).createObjectURL = vi.fn(() => "blob:fake");
  (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = vi.fn();
});
afterEach(() => vi.restoreAllMocks());

describe("FocusClaims (via ?focus=claims)", () => {
  it("deep-links the report and source panes to the selected claim's pages", async () => {
    const blob = mockRun();

    renderAt("/runs/r?focus=claims");
    // Generous timeouts: under the full parallel suite this render can take
    // well over the 1s default.
    const reportFrame = await screen.findByTitle("report", {}, { timeout: 5000 });
    expect(reportFrame).toHaveAttribute("src", "blob:fake#page=1&toolbar=0&navpanes=0&view=FitH");
    const sourceFrame = await screen.findByTitle("source", {}, { timeout: 5000 });
    expect(sourceFrame).toHaveAttribute("src", "blob:fake#page=3&toolbar=0&navpanes=0&view=FitH");
    expect(blob).toHaveBeenCalledWith("r", "reportdoc");
    expect(blob).toHaveBeenCalledWith("r", "sourcedoc");
    // The selected claim's rationale and quote sit in the strip below (the
    // fixture's claim text equals its quote, so both render "“hunger fell”").
    expect(screen.getByText("verbatim")).toBeInTheDocument();
    expect(screen.getAllByText("“hunger fell”").length).toBeGreaterThan(0);
  }, 15000);

  it("honors verdict and claim params, and Close returns to the panels", async () => {
    mockRun();

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

  it("cites each source in the claim list by its own kind of locator", async () => {
    mockRun();
    renderAt("/runs/r?focus=claims");
    await waitFor(() => expect(screen.getByText("Src · p.3")).toBeInTheDocument());
    expect(screen.getByText("Water in Crisis · § Findings")).toBeInTheDocument();
    // An untitled video is named by its link and cited by its start time.
    expect(screen.getByText("www.youtube.com/watch · 1:02:03")).toBeInTheDocument();
    expect(screen.getByText("No source coverage")).toBeInTheDocument();
  });

  it("opens a web source as readable text with the quote marked", async () => {
    const blob = mockRun();
    const json = vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);

    renderAt("/runs/r?focus=claims&claim=c3");
    expect(await screen.findByRole("heading", { name: "Water in Crisis" })).toBeInTheDocument();
    await waitFor(() =>
      expect(document.querySelector(".readable mark")?.textContent).toBe(
        "Two billion people lack safe water"
      )
    );
    expect(json).toHaveBeenCalledWith("r", "webdoc");
    expect(blob).not.toHaveBeenCalledWith("r", "webdoc");
    expect(screen.queryByTitle("source")).not.toBeInTheDocument();
    // The pane header names the page and its section.
    expect(screen.getByText(/· Water in Crisis · § Findings/)).toBeInTheDocument();
  });

  it("offers a video's original instead of a preview", async () => {
    mockRun();
    renderAt("/runs/r?focus=claims&claim=c4");
    expect(
      await screen.findByText(/preview isn't available for this video yet/)
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open original ↗" })).toHaveAttribute(
      "href",
      "https://www.youtube.com/watch?v=abc123def45"
    );
  });
});
