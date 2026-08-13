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
  postChat
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

export function useRun(runId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.run(runId ?? ""),
    queryFn: () => getRun(runId as string),
    enabled: Boolean(runId),
    refetchInterval: (query: { state: { data?: RunDetail } }) =>
      isTerminal(query.state.data?.run.status) ? false : POLL_MS
  });
}

export function useReport(runId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.report(runId ?? ""),
    queryFn: () => getReport(runId as string),
    enabled: Boolean(runId),
    refetchInterval: (query: { state: { data?: Report } }) =>
      isTerminal(query.state.data?.status) ? false : POLL_MS
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
