import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useRuns } from "./queries";
import * as v2 from "./v2";

function makeWrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

function RunsList() {
  const { data, isLoading } = useRuns();
  if (isLoading) return <p>loading</p>;
  return (
    <ul>
      {data?.map((run) => (
        <li key={run.id}>{run.status}</li>
      ))}
    </ul>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("useRuns", () => {
  it("renders the fetched runs under a QueryClientProvider", async () => {
    vi.spyOn(v2, "listRuns").mockResolvedValue([
      { id: "r1", status: "DONE", created_at: "t", error: null, title: null, source_count: null, scores: null }
    ]);

    render(<RunsList />, { wrapper: makeWrapper() });
    await waitFor(() => expect(screen.getByText("DONE")).toBeInTheDocument());
  });

  it("polls while a run is non-terminal, then stops once all are terminal", async () => {
    const running = [
      {
        id: "r1",
        status: "RUNNING" as const,
        created_at: "t",
        error: null,
        title: null,
        source_count: null,
        scores: null
      }
    ];
    const spy = vi
      .spyOn(v2, "listRuns")
      .mockResolvedValueOnce(running)
      .mockResolvedValue([{ ...running[0], status: "DONE" as const }]);

    render(<RunsList />, { wrapper: makeWrapper() });
    await waitFor(() => expect(screen.getByText("RUNNING")).toBeInTheDocument());
    // The RUNNING payload schedules a refetch; the DONE payload must stop it.
    await waitFor(() => expect(screen.getByText("DONE")).toBeInTheDocument(), {
      timeout: 4000
    });
    const callsAfterDone = spy.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 2000));
    expect(spy.mock.calls.length).toBe(callsAfterDone);
  }, 10000);
});
