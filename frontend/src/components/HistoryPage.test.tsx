import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as v2 from "../api/v2";
import HistoryPage from "./HistoryPage";

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <HistoryPage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("HistoryPage", () => {
  it("lists runs with a status label and links to the run", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue([
      { id: "abcdef1234567890", status: "DONE", created_at: "2026-08-13T10:00:00Z", error: null },
      { id: "0987654321fedcba", status: "RUNNING", created_at: "2026-08-13T09:00:00Z", error: null }
    ]);

    renderPage();
    await waitFor(() => expect(screen.getByText("Complete")).toBeInTheDocument());
    expect(screen.getByText("Running")).toBeInTheDocument();
    // Short id shown, and each row links to the run page.
    expect(screen.getByText("abcdef12")).toBeInTheDocument();
    const link = screen.getByText("abcdef12").closest("a");
    expect(link).toHaveAttribute("href", "/runs/abcdef1234567890");
  });

  it("shows an empty state when there are no runs", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue([]);
    renderPage();
    await waitFor(() => expect(screen.getByText(/No runs yet/)).toBeInTheDocument());
  });
});
