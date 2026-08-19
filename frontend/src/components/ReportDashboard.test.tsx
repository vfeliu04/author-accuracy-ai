import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Report, RunDetail } from "../api/types";
import * as v2 from "../api/v2";
import ReportDashboard from "./ReportDashboard";

function renderAt(runId: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/runs/${runId}`]}>
        <Routes>
          <Route path="/runs/:runId" element={<ReportDashboard />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

const runningDetail: RunDetail = {
  run: { id: "r", status: "RUNNING", created_at: "t", error: null },
  job: {
    id: "j",
    run_id: "r",
    kind: "full_pipeline",
    status: "RUNNING",
    payload: { report_upload_id: "", source_upload_ids: [] },
    progress: [
      { step: "ingest", label: "Ingesting documents", status: "done", ts: "t" },
      { step: "extract", label: "Extracting claims", status: "running", ts: "t" }
    ],
    error: null,
    created_at: "t",
    updated_at: "t"
  }
};

const doneReport: Report = {
  run_id: "r",
  status: "DONE",
  report_doc_id: "d",
  scores: { accuracy: 1, coverage: 0.5, credibility: 0.8, validity: 0.6 },
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
  sources: [{ doc_id: "s", title: "Src", total: 80, tier: "VERIFIED_DOI", components: {} }]
};

afterEach(() => vi.restoreAllMocks());

describe("ReportDashboard", () => {
  it("shows the pipeline progress feed while the run is not done", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue(runningDetail);
    vi.spyOn(v2, "getReport").mockResolvedValue({ ...doneReport, status: "RUNNING", scores: null });
    renderAt("r");
    await waitFor(() => expect(screen.getByText("Extracting claims")).toBeInTheDocument());
    expect(screen.getByText("Running the pipeline")).toBeInTheDocument();
  });

  it("offers a retry on a failed run and calls the endpoint", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue({
      run: { id: "r", status: "FAILED", created_at: "t", error: "Connection error." },
      job: null
    });
    vi.spyOn(v2, "getReport").mockResolvedValue({ ...doneReport, status: "FAILED", scores: null });
    const retry = vi
      .spyOn(v2, "retryRun")
      .mockResolvedValue({ run_id: "r", job_id: "j", status: "QUEUED" });
    renderAt("r");
    await waitFor(() => expect(screen.getByText(/This run failed/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Retry run" }));
    await waitFor(() => expect(retry).toHaveBeenCalledWith("r"));
  });

  it("shows claims, sources, and rating when the run is done", async () => {
    vi.spyOn(v2, "getRun").mockResolvedValue({
      run: { id: "r", status: "DONE", created_at: "t", error: null },
      job: null
    });
    vi.spyOn(v2, "getReport").mockResolvedValue(doneReport);
    renderAt("r");
    await waitFor(() => expect(screen.getByText("hunger fell")).toBeInTheDocument());
    expect(screen.getByText("SUPPORTED")).toBeInTheDocument();
    expect(screen.getByText("Rating")).toBeInTheDocument();
    expect(screen.getByText("VERIFIED_DOI")).toBeInTheDocument();
  });
});
