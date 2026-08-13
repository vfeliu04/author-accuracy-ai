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
      { id: "r1", status: "DONE", created_at: "t", error: null }
    ]);

    render(<RunsList />, { wrapper: makeWrapper() });
    await waitFor(() => expect(screen.getByText("DONE")).toBeInTheDocument());
  });
});
