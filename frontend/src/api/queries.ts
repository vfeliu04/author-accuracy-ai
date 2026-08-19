// TanStack Query hooks over the v2 fetchers. Polling stops on terminal status
// so a finished run isn't refetched forever; mutations invalidate the run list.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { ChatMode, ChatTurn, RunDetail, Report } from "./types";
import { isTerminal } from "./types";
import {
  createRun,
  fetchPdfBlob,
  getReport,
  getRun,
  listRuns,
  postChat,
  retryRun
} from "./v2";

const POLL_MS = 1500;

export const queryKeys = {
  runs: ["runs"] as const,
  run: (runId: string) => ["run", runId] as const,
  report: (runId: string) => ["report", runId] as const,
  pdf: (runId?: string, docId?: string | null) => ["pdf", runId, docId] as const
};

export function useRuns() {
  return useQuery({ queryKey: queryKeys.runs, queryFn: listRuns });
}

// Stop polling once the run reaches a terminal status OR the query itself
// errored — an error state carries no data, so a status-only check would poll
// a 404'd /report forever.
function pollUntilTerminal(status: string, isTerminalData: boolean): number | false {
  return status === "error" || isTerminalData ? false : POLL_MS;
}

export function useRun(runId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.run(runId ?? ""),
    queryFn: () => getRun(runId as string),
    enabled: Boolean(runId),
    refetchInterval: (query: { state: { status: string; data?: RunDetail } }) =>
      pollUntilTerminal(query.state.status, isTerminal(query.state.data?.run.status))
  });
}

export function useReport(runId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.report(runId ?? ""),
    queryFn: () => getReport(runId as string),
    enabled: Boolean(runId),
    refetchInterval: (query: { state: { status: string; data?: Report } }) =>
      pollUntilTerminal(query.state.status, isTerminal(query.state.data?.status))
  });
}

export function useCreateRun() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ report, sources }: { report: File; sources: File[] }) =>
      createRun(report, sources),
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.runs })
  });
}

export function useChat(runId: string) {
  return useMutation({
    mutationFn: (body: { question: string; history: ChatTurn[]; mode: ChatMode }) =>
      postChat(runId, body)
  });
}

// Retrying flips the run back to RUNNING server-side; invalidating the run
// query restarts the polling that stopped on the terminal FAILED status.
export function useRetryRun(runId: string | undefined) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => retryRun(runId as string),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: queryKeys.run(runId ?? "") });
      client.invalidateQueries({ queryKey: queryKeys.report(runId ?? "") });
      client.invalidateQueries({ queryKey: queryKeys.runs });
    }
  });
}

// Fetches a run's PDF as an authenticated blob (an iframe can't send the API
// key) and hands back an object URL, revoking the previous one on change.
export function usePdfBlob(runId?: string, docId?: string | null) {
  const query = useQuery({
    queryKey: queryKeys.pdf(runId, docId),
    queryFn: () => fetchPdfBlob(runId as string, docId as string),
    enabled: Boolean(runId && docId),
    staleTime: Infinity, // a run's stored PDF never changes
    gcTime: Infinity
  });

  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!query.data) {
      setUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(query.data);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [query.data]);

  return { url, isLoading: query.isLoading, error: query.error as Error | null };
}
