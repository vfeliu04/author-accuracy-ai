import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { queryKeys, useDeleteRun, useReferenceScan, useRuns, useSnapshot } from "./queries";
import * as v2 from "./v2";

function makeWrapper(client = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
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

describe("useDeleteRun", () => {
  it("forgets every stored file of the deleted run, readable pages included", async () => {
    vi.spyOn(v2, "deleteRun").mockResolvedValue(undefined);
    vi.spyOn(v2, "listRuns").mockResolvedValue([]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(queryKeys.pdf("r1", "d1"), new Blob(["%PDF"]));
    client.setQueryData(queryKeys.snapshot("r1", "d2"), { schema: 1 });
    client.setQueryData(queryKeys.snapshot("r2", "d3"), { schema: 1 });

    const { result } = renderHook(() => useDeleteRun(), { wrapper: makeWrapper(client) });
    await act(() => result.current.mutateAsync("r1"));

    expect(client.getQueryData(queryKeys.pdf("r1", "d1"))).toBeUndefined();
    expect(client.getQueryData(queryKeys.snapshot("r1", "d2"))).toBeUndefined();
    // Another run's pages stay cached.
    expect(client.getQueryData(queryKeys.snapshot("r2", "d3"))).toEqual({ schema: 1 });
  });
});

describe("useSnapshot", () => {
  it("reads a run's stored page once, and not again when the page is shown again", async () => {
    const read = vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      schema: 1,
      document: { title: "Water in Crisis", sections: [{ title: "", page: null, text: "Body" }] },
      provenance: { url: "https://example.org/w", final_url: "https://example.org/w" }
    });
    const wrapper = makeWrapper();

    const first = renderHook(() => useSnapshot("r1", "d1"), { wrapper });
    await waitFor(() => expect(first.result.current.isSuccess).toBe(true));
    first.unmount();

    const second = renderHook(() => useSnapshot("r1", "d1"), { wrapper });
    expect(second.result.current.data?.document.title).toBe("Water in Crisis");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(read).toHaveBeenCalledTimes(1);
  });
});

describe("useReferenceScan", () => {
  // Every other test here turns retries off for the whole client, which
  // would hide the hook's own setting. This client keeps TanStack's default
  // (three retries) and only removes the backoff, so a hook that retried
  // would fail in milliseconds with four calls, not after seven seconds.
  it("asks once and never retries: a refusal names the file, which won't change", async () => {
    const scan = vi
      .spyOn(v2, "scanReferences")
      .mockRejectedValue(new Error("“report.pdf” is not a PDF."));
    const client = new QueryClient({ defaultOptions: { queries: { retryDelay: 0 } } });
    const file = new File([new Uint8Array(4)], "report.pdf", { type: "application/pdf" });

    const { result } = renderHook(() => useReferenceScan(file), { wrapper: makeWrapper(client) });
    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error?.message).toBe("“report.pdf” is not a PDF.");
    expect(scan).toHaveBeenCalledTimes(1);
  });

  it("does nothing without a report", () => {
    const scan = vi.spyOn(v2, "scanReferences");
    const { result } = renderHook(() => useReferenceScan(null), { wrapper: makeWrapper() });
    expect(result.current.fetchStatus).toBe("idle");
    expect(scan).not.toHaveBeenCalled();
  });
});
