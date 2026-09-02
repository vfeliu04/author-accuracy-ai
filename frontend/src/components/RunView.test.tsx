import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Report, RunDetail } from "../api/types";
import * as v2 from "../api/v2";
import RunView from "./RunView";

function renderAt(runId: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/runs/${runId}`]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunView />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

const runningDetail: RunDetail = {
  run: {
    id: "r",
    status: "RUNNING",
    created_at: "t",
    error: null,
    title: "Coastal Brief",
    source_count: 1,
    scores: null
  },
  job: {
    id: "j",
    run_id: "r",
    kind: "full_pipeline",
    status: "RUNNING",
    payload: { report_upload_id: "u1", source_upload_ids: ["u2"] },
    progress: [
      { step: "ingest", label: "Ingested 2 documents", status: "done", ts: "t" },
      { step: "extract", label: "Extracting claims", status: "running", ts: "t" }
    ],
    error: null,
    created_at: "t",
    updated_at: "t"
  },
  uploads: [
    { id: "u1", kind: "REPORT", file_name: "coastal_brief.pdf" },
    { id: "u2", kind: "SOURCE", file_name: "ipcc_ch3.pdf" }
  ]
};

const doneReport: Report = {
  run_id: "r",
  title: "Coastal Brief",
  status: "DONE",
  report_doc_id: "d",
  scores: { accuracy: 1, coverage: 0.5, credibility: 0.8, validity: 0.6 },
  accuracy_detail: null,
  validity_detail: null,
  credibility_detail: null,
  stats: { claims_total: 1, claims_supported: 1, claims_contradicted: 0, claims_unverifiable: 0 },
  claims: [
    {
      claim_id: "c",
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
      rationale: "stated verbatim",
      year_flag: null,
      evidence_source: { doc_id: "s", title: "Src", page: 3 }
    }
  ],
  sources: [{ doc_id: "s", title: "Src", total: 80, tier: "VERIFIED_DOI", components: {}, metadata: {} }]
};

afterEach(() => vi.restoreAllMocks());

describe("RunView", () => {
  it("shows progress, upload filenames, and a locked chat while running", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue(runningDetail);
    vi.spyOn(v2, "getReport").mockResolvedValue({
      ...doneReport,
      status: "RUNNING",
      scores: null,
      stats: { claims_total: 0, claims_supported: 0, claims_contradicted: 0, claims_unverifiable: 0 },
      claims: [],
      sources: []
    });
    renderAt("r");
    await waitFor(() =>
      expect(screen.getByText("Verifying this report against its sources")).toBeInTheDocument()
    );
    // Client vocabulary for unfinished steps; server result string once done.
    expect(screen.getByText("Extract claims")).toBeInTheDocument();
    expect(screen.getByText("Ingested 2 documents")).toBeInTheDocument();
    // Sources panel shows the uploaded filenames before scoring exists.
    expect(screen.getByText("ipcc_ch3.pdf")).toBeInTheDocument();
    expect(screen.getByText("coastal_brief.pdf")).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("Chat unlocks when verification completes…")
    ).toBeDisabled();
    // Analysis rings sit in ghost mode.
    expect(screen.getByText("Scores, claims and chat appear here when verification completes.")).toBeInTheDocument();
  });

  it("offers a retry on a failed run and calls the endpoint", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue({
      ...runningDetail,
      run: { ...runningDetail.run, status: "FAILED", error: "Connection error." }
    });
    vi.spyOn(v2, "getReport").mockResolvedValue({ ...doneReport, status: "FAILED", scores: null });
    const retry = vi
      .spyOn(v2, "retryRun")
      .mockResolvedValue({ run_id: "r", job_id: "j", status: "QUEUED" });
    renderAt("r");
    await waitFor(() =>
      expect(screen.getByText("This verification failed")).toBeInTheDocument()
    );
    // The raw error is translated for humans.
    expect(screen.getByText(/laptop sleep or dropped Wi-Fi/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry run" }));
    await waitFor(() => expect(retry).toHaveBeenCalledWith("r"));
  });

  it("shows a report-load error with a retry affordance instead of a stuck progress feed", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue({
      ...runningDetail,
      run: { ...runningDetail.run, status: "DONE" }
    });
    vi.spyOn(v2, "getReport").mockRejectedValue(new Error("database is locked"));
    renderAt("r");
    await waitFor(
      () => expect(screen.getByText(/Could not load this run/)).toBeInTheDocument(),
      { timeout: 4000 }
    );
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(
      screen.queryByText("Verifying this report against its sources")
    ).not.toBeInTheDocument();
  }, 10000);

  it("keeps chat errors out of the message history", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue({
      ...runningDetail,
      run: { ...runningDetail.run, status: "DONE" }
    });
    vi.spyOn(v2, "getReport").mockResolvedValue(doneReport);
    vi.spyOn(v2, "postChat").mockRejectedValue(new Error("chat service down"));
    renderAt("r");
    const input = await screen.findByPlaceholderText("Ask about this verification…");
    fireEvent.change(input, { target: { value: "why?" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(screen.getByText(/chat service down/)).toBeInTheDocument());
    // The user's turn renders; no fabricated assistant reply joins the history.
    expect(screen.getByText("why?")).toBeInTheDocument();
    expect(document.querySelectorAll(".msg--assistant")).toHaveLength(0);
  });

  it("shows sources with credibility, chat, and rings together when done", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue({
      ...runningDetail,
      run: { ...runningDetail.run, status: "DONE" }
    });
    vi.spyOn(v2, "getReport").mockResolvedValue(doneReport);
    renderAt("r");
    await waitFor(() => expect(screen.getByText("Src")).toBeInTheDocument());
    // Credibility badge + tier on the source row ("80" also appears in the
    // credibility ring — both are correct).
    expect(screen.getAllByText("80").length).toBeGreaterThan(0);
    expect(screen.getByText("verified DOI")).toBeInTheDocument();
    // Chat is live and the rings are up — all in one view.
    expect(screen.getByPlaceholderText("Ask about this verification…")).toBeEnabled();
    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    // The claim count shows on the report row and the Claims tile; no
    // composite "Overall" anywhere.
    expect(screen.getAllByText("1 claim").length).toBeGreaterThan(0);
    expect(screen.queryByText(/overall/i)).not.toBeInTheDocument();
  });
});
