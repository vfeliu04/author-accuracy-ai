import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunListItem } from "../api/types";
import * as v2 from "../api/v2";
import HomePage from "./HomePage";

const RUNS: RunListItem[] = [
  {
    id: "done1234deadbeef",
    status: "DONE",
    created_at: "2026-08-21T10:00:00Z",
    error: null,
    title: "Water Stress Report",
    source_count: 6,
    scores: { accuracy: 0.9, coverage: 0.57, credibility: 0.72, validity: 0.45 }
  },
  {
    id: "run5678cafebabe0",
    status: "RUNNING",
    created_at: "2026-08-22T10:00:00Z",
    error: null,
    title: "Coastal Flooding Brief",
    source_count: 4,
    scores: null
  },
  {
    id: "fail9012aabbccdd",
    status: "FAILED",
    created_at: "2026-08-20T10:00:00Z",
    error: "Connection error.",
    title: null,
    source_count: 3,
    scores: null
  }
];

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("HomePage", () => {
  it("renders cards with title, meta, and score pills", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue(RUNS);
    renderPage();
    await waitFor(() => expect(screen.getByText("Water Stress Report")).toBeInTheDocument());
    expect(screen.getByText("A 90")).toBeInTheDocument();
    expect(screen.getByText("C 72")).toBeInTheDocument();
    expect(screen.getByText("V 45")).toBeInTheDocument();
    expect(screen.getByText(/6 sources/)).toBeInTheDocument();
    // The title-less failed run falls back to a short id.
    expect(screen.getByText("Run fail9012")).toBeInTheDocument();
  });

  it("filters by status chips", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue(RUNS);
    renderPage();
    await waitFor(() => expect(screen.getByText("Water Stress Report")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Failed" }));
    expect(screen.queryByText("Water Stress Report")).not.toBeInTheDocument();
    expect(screen.getByText("Run fail9012")).toBeInTheDocument();
  });

  it("retries a failed run from its card", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue(RUNS);
    const retry = vi
      .spyOn(v2, "retryRun")
      .mockResolvedValue({ run_id: "fail9012aabbccdd", job_id: "j", status: "QUEUED" });
    renderPage();
    await waitFor(() => expect(screen.getByText("Run fail9012")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "↻ Retry" }));
    await waitFor(() => expect(retry).toHaveBeenCalledWith("fail9012aabbccdd"));
  });

  it("surfaces a failed retry on the card instead of failing silently", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue(RUNS);
    vi.spyOn(v2, "retryRun").mockRejectedValue(new Error("has no job to retry"));
    renderPage();
    await waitFor(() => expect(screen.getByText("Run fail9012")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "↻ Retry" }));
    await waitFor(() => expect(screen.getByText(/has no job to retry/)).toBeInTheDocument());
  });

  it("deletes a run through the kebab menu with confirmation", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue(RUNS);
    const del = vi.spyOn(v2, "deleteRun").mockResolvedValue(undefined);
    renderPage();
    await waitFor(() => expect(screen.getByText("Water Stress Report")).toBeInTheDocument());
    fireEvent.click(screen.getAllByRole("button", { name: "Run options" })[0]);
    fireEvent.click(screen.getByRole("button", { name: "Delete verification…" }));
    // The confirmation names the run and requires an explicit click.
    expect(screen.getByRole("dialog", { name: "Delete verification" })).toBeInTheDocument();
    expect(del).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete verification" }));
    await waitFor(() => expect(del).toHaveBeenCalledWith("done1234deadbeef"));
  });

  it("opens the upload dialog from the create card", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue([]);
    renderPage();
    await waitFor(() => expect(screen.getByText(/No verifications yet/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Start a new verification" }));
    expect(screen.getByRole("dialog", { name: "New verification" })).toBeInTheDocument();
  });
});
