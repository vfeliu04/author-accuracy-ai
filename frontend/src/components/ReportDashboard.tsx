import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import InternalSourcesPanel from "./InternalSourcesPanel";
import SummaryPanel from "./SummaryPanel";
import ChatPanel, { type ChatMessage } from "./ChatPanel";
import RatingPanel from "./RatingPanel";
import AnalyticsPanel from "./AnalyticsPanel";
import ClaimsPanel from "./ClaimsPanel";
import ClaimsWorkspace from "./ClaimsWorkspace";
import { useReportData } from "../context/ReportDataContext";
import {
  getLatestReportSummary,
  getReportSummary,
  getReportClaims,
  sendChat,
  getChatHistory,
  type ChatMode,
  type ClaimSummary
} from "../api/client";

const ACTIVE_JOB_STORAGE_KEY = "active_job_id";

function formatReportTitle(filename: string): string {
  return filename
    .replace(/\.pdf$/i, "")
    .replace(/[_\-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

const CHAT_SUGGESTIONS = [
  "Which claims are contradicted?",
  "What is the weakest supported claim?",
  "How can I improve my accuracy score?",
  "Which source backs the most claims?",
];

// ReportDashboard represents the full page layout for the report quality view.
const ReportDashboard = () => {
  const navigate = useNavigate();
  const {
    internalSources,
    addInternalSource,
    reportDocument,
    summaryData,
    setSummaryData,
    getChatMessages,
    setChatMessages,
    jobStatus,
    refreshJobStatus
  } = useReportData();
  const [reportTitle, setReportTitle] = useState("Report Dashboard");
  const [summary, setSummary] = useState("Upload a report to view results.");
  const [scores, setScores] = useState({ overall: 0, accuracy: 0, credibility: 0, validity: 0 });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatMode, setChatMode] = useState<ChatMode>("evidence");
  const [modeLocked, setModeLocked] = useState(false);
  const [modeSuggestion, setModeSuggestion] = useState<ChatMode | null>(null);
  const [claimsList, setClaimsList] = useState<ClaimSummary[]>([]);
  const [claimsTotal, setClaimsTotal] = useState(0);
  const [claimsHasMore, setClaimsHasMore] = useState(false);
  const [claimsPage, setClaimsPage] = useState(0);
  const [claimsWorkspaceOpen, setClaimsWorkspaceOpen] = useState(false);
  const sessionIdRef = useRef(`sess-${Date.now()}`);
  const isMountedRef = useRef(true);

  const applySummary = useCallback(
    async (data: Awaited<ReturnType<typeof getLatestReportSummary>>) => {
      if (!isMountedRef.current) {
        return;
      }
      localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, data.job_id);
      setSummaryData(data);
      await refreshJobStatus(data.job_id);
      try {
        const claimsData = await getReportClaims(data.job_id, 0, 10);
        setClaimsList(claimsData.claims);
        setClaimsTotal(claimsData.total);
        setClaimsHasMore(claimsData.has_more);
        setClaimsPage(0);
      } catch {
        // claims fetch failure should not block dashboard
      }
    },
    [refreshJobStatus, setSummaryData]
  );

  const fetchDashboard = useCallback(
    async (jobIdOverride?: string) => {
      setLoading(true);
      setError(null);
      const storedJobId = jobIdOverride ?? localStorage.getItem(ACTIVE_JOB_STORAGE_KEY) ?? undefined;
      try {
        if (storedJobId) {
          const data = await getReportSummary(storedJobId);
          await applySummary(data);
        } else {
          const data = await getLatestReportSummary();
          await applySummary(data);
        }
      } catch (primaryError) {
        if (!jobIdOverride && storedJobId) {
          try {
            const fallback = await getLatestReportSummary();
            await applySummary(fallback);
          } catch (fallbackError) {
            if (isMountedRef.current) {
              setError(
                fallbackError instanceof Error
                  ? fallbackError.message
                  : "Failed to load dashboard data."
              );
            }
          }
        } else if (isMountedRef.current) {
          setError(
            primaryError instanceof Error ? primaryError.message : "Failed to load dashboard data."
          );
        }
      } finally {
        if (isMountedRef.current) {
          setLoading(false);
        }
      }
    },
    [applySummary]
  );

  useEffect(() => {
    isMountedRef.current = true;
    fetchDashboard().catch(() => undefined);
    return () => {
      isMountedRef.current = false;
    };
  }, [fetchDashboard]);

  useEffect(() => {
    if (!summaryData) {
      return;
    }
    setChatMode("evidence");
    setModeLocked(false);
    setModeSuggestion(null);
    setReportTitle(formatReportTitle(summaryData.report.name));
    setSummary(summaryData.report.summary);
    setScores(summaryData.scores);
    const cachedHistory = getChatMessages(summaryData.job_id);
    if (cachedHistory.length) {
      setMessages(
        cachedHistory.map((entry, index) => ({
          id: index + 1,
          author: entry.role.toLowerCase() === "assistant" ? "System" : "User",
          text: entry.message
        }))
      );
    } else {
      const accuracyDisplay = `${Math.round(summaryData.scores.accuracy * 100)}%`;
      const introText = `Summary: ${summaryData.report.summary || "No summary available."}\nAccuracy: ${accuracyDisplay} of evaluated claims are currently supported.`;
      setMessages([{ id: Date.now(), author: "System", text: introText }]);
    }
  }, [summaryData, getChatMessages]);

  useEffect(() => {
    if (!summaryData) {
      return;
    }
    const { job_id: jobId, scores, report } = summaryData;
    const cachedHistory = getChatMessages(jobId);
    async function fetchDetail() {
      try {
        const [{ history }] = await Promise.all([
          getChatHistory(jobId)
        ]);
        if (history.length) {
          setChatMessages(jobId, history);
          setMessages(
            history.map((entry, index) => ({
              id: index + 1,
              author: entry.role.toLowerCase() === "assistant" ? "System" : "User",
              text: entry.message
            }))
          );
        } else if (!cachedHistory.length) {
          const accuracyDisplay = `${Math.round(scores.accuracy * 100)}%`;
          const introText = `Summary: ${report.summary || "No summary available."}\nAccuracy: ${accuracyDisplay} of evaluated claims are supported.`;
          setMessages([{ id: Date.now(), author: "System", text: introText }]);
        }
      } catch (fetchError) {
        console.error(fetchError);
      }
    }
    fetchDetail();
  }, [summaryData, setChatMessages]);

  const handleSourceSelect = (sourceId: string) => {
    navigate(`/dashboard/sources/${sourceId}`);
  };

  const handleRefreshSummary = useCallback(() => {
    const targetJobId = jobStatus?.job_id ?? summaryData?.job_id ?? undefined;
    fetchDashboard(targetJobId).catch(() => undefined);
  }, [fetchDashboard, jobStatus, summaryData]);

  const handleSendMessage = async (text: string) => {
    if (!summaryData) {
      setError("Report data not loaded yet.");
      return;
    }
    const jobId = summaryData.job_id;
    const timestamp = Date.now();
    const userMessage: ChatMessage = { id: timestamp, author: "User", text };
    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);
    try {
      const response = await sendChat(
        text,
        jobId,
        sessionIdRef.current,
        chatMode,
        modeLocked
      );
      setMessages((prev) => [
        ...prev,
        { id: timestamp + 1, author: "System", text: response.answer }
      ]);
      if (!modeLocked) {
        setChatMode(response.mode);
      }
      if (!modeLocked && response.suggested_mode) {
        setModeSuggestion(response.suggested_mode);
      } else if (!response.suggested_mode) {
        setModeSuggestion(null);
      }
      const historyEntryUser = {
        session_id: sessionIdRef.current,
        role: "user",
        message: text,
        timestamp: new Date().toISOString(),
        context_ids: {}
      };
      const historyEntryAssistant = {
        session_id: sessionIdRef.current,
        role: "assistant",
        message: response.answer,
        timestamp: new Date().toISOString(),
        context_ids: {
          mode: response.mode,
          claims: response.claims_used.map((claim) => claim.claim_id),
          sources: response.sources_used
            .map((source) => source.source_id)
            .filter((value): value is string => Boolean(value))
        }
      };
      setChatMessages(jobId, [
        ...getChatMessages(jobId),
        historyEntryUser,
        historyEntryAssistant
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: timestamp + 1,
          author: "System",
          text: "The chat service is currently unavailable. Please try again later."
        }
      ]);
      setError(err instanceof Error ? err.message : "The chat service is currently unavailable.");
    } finally {
      setIsSending(false);
    }
  };

  const handleModeChange = (nextMode: ChatMode) => {
    setChatMode(nextMode);
    setModeLocked(true);
    setModeSuggestion(null);
  };

  const handleModeReset = () => {
    setModeLocked(false);
    setModeSuggestion(null);
  };

  const handleSuggestionAccept = (suggested: ChatMode) => {
    setChatMode(suggested);
    setModeLocked(true);
    setModeSuggestion(null);
  };

  const handleSuggestionDismiss = () => {
    setModeSuggestion(null);
  };

  const handleLoadMoreClaims = useCallback(async () => {
    if (!summaryData) return;
    const nextPage = claimsPage + 1;
    try {
      const claimsData = await getReportClaims(summaryData.job_id, nextPage, 10);
      setClaimsList((prev) => [...prev, ...claimsData.claims]);
      setClaimsTotal(claimsData.total);
      setClaimsHasMore(claimsData.has_more);
      setClaimsPage(nextPage);
    } catch {
      // silently fail
    }
  }, [summaryData, claimsPage]);

  const handleOpenWorkspace = useCallback(async () => {
    // Fetch all claims if pagination is incomplete
    if (claimsHasMore && summaryData) {
      try {
        const all = await getReportClaims(summaryData.job_id, 0, Math.max(claimsTotal, 500));
        setClaimsList(all.claims);
        setClaimsTotal(all.total);
        setClaimsHasMore(all.has_more);
      } catch {
        // proceed with what we have
      }
    }
    setClaimsWorkspaceOpen(true);
  }, [claimsHasMore, claimsTotal, summaryData]);

  const evaluatedSources = useMemo(() => {
    if (!summaryData) {
      return internalSources;
    }
    return summaryData.sources.map((source) => ({
      id: source.id,
      name: source.name,
      filePath: source.file_url ?? "",
      summary: source.summary,
      usageCount: source.usage_count,
      scores: {
        credibility: source.scores.credibility,
        validity: source.scores.credibility,
        overall: source.scores.credibility
      }
    }));
  }, [internalSources, summaryData]);

  const formatTimestamp = useCallback((value?: string | null) => {
    if (!value) {
      return null;
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return null;
    }
    return parsed.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }, []);

  const statusLabel = useMemo(() => {
    if (!jobStatus) {
      return null;
    }
    switch (jobStatus.status) {
      case "RUNNING":
        return "Running";
      case "QUEUED":
        return "Queued";
      case "FAILED":
        return "Failed";
      case "DONE":
        return "Complete";
      default:
        return jobStatus.status;
    }
  }, [jobStatus]);

  const badgeClass = useMemo(() => {
    if (!jobStatus) {
      return null;
    }
    switch (jobStatus.status) {
      case "RUNNING":
        return "dashboard__badge dashboard__badge--running";
      case "QUEUED":
        return "dashboard__badge dashboard__badge--queued";
      case "FAILED":
        return "dashboard__badge dashboard__badge--failed";
      case "DONE":
        return "dashboard__badge dashboard__badge--done";
      default:
        return "dashboard__badge dashboard__badge--idle";
    }
  }, [jobStatus]);

  const lastRunText = useMemo(() => {
    if (jobStatus?.status === "DONE") {
      const formatted = formatTimestamp(jobStatus.updated_at);
      return formatted ? `Last run at ${formatted}` : "Last run complete.";
    }
    if (jobStatus?.status === "FAILED") {
      const formatted = formatTimestamp(jobStatus.updated_at);
      return formatted ? `Pipeline failed at ${formatted}` : "Pipeline failed.";
    }
    if (jobStatus && jobStatus.updated_at) {
      const formatted = formatTimestamp(jobStatus.updated_at);
      return formatted ? `Pipeline running (started ${formatted})` : "Pipeline running…";
    }
    return summaryData ? "Report ready — awaiting pipeline status." : "No pipeline runs yet.";
  }, [jobStatus, summaryData, formatTimestamp]);

  const showRefreshButton =
    jobStatus?.status === "DONE" && summaryData?.job_id !== jobStatus.job_id;

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>{reportTitle}</h1>
          {summaryData && (
            <p className="dashboard__subtitle">
              {summaryData.stats.claims_supported} of {summaryData.stats.claims_total} claims supported
              &nbsp;·&nbsp;{summaryData.sources.length} source{summaryData.sources.length !== 1 ? "s" : ""}
              &nbsp;·&nbsp;{Math.round(summaryData.scores.accuracy * 100)}% accuracy
            </p>
          )}
          {error ? <p className="dashboard__status dashboard__status--error">{error}</p> : null}
          {!error && loading ? (
            <p className="dashboard__status">Loading latest metrics…</p>
          ) : null}
        </div>
        <div className="dashboard__meta">
          <span className="dashboard__meta-time">{lastRunText}</span>
          {badgeClass && statusLabel ? <span className={badgeClass}>{statusLabel}</span> : null}
          {showRefreshButton ? (
            <button
              type="button"
              className="dashboard__refresh-button"
              onClick={handleRefreshSummary}
              disabled={loading}
            >
              Refresh results
            </button>
          ) : null}
        </div>
      </header>
      <main className="dashboard__content">
        <section className="dashboard__column dashboard__column--left dashboard__column--stacked-left">
          <InternalSourcesPanel
            sources={evaluatedSources}
            onSelectSource={handleSourceSelect}
            onAddSource={addInternalSource}
          />
          <SummaryPanel
            summary={summaryData?.report.summary ?? summary}
            reportLabel={reportDocument?.name}
            onOpenReport={reportDocument ? () => navigate(`/dashboard/report`) : undefined}
            stats={summaryData?.stats}
          />
          {claimsList.length > 0 && (
            <ClaimsPanel
              claims={claimsList}
              totalClaims={claimsTotal}
              hasMore={claimsHasMore}
              onLoadMore={handleLoadMoreClaims}
              onExpand={handleOpenWorkspace}
            />
          )}
          {claimsWorkspaceOpen && summaryData && (
            <ClaimsWorkspace
              claims={claimsList}
              hasMore={claimsHasMore}
              onLoadMore={handleLoadMoreClaims}
              reportUploadId={summaryData.report.id}
              onClose={() => setClaimsWorkspaceOpen(false)}
            />
          )}
        </section>
        <section className="dashboard__column dashboard__column--center">
          <ChatPanel
            messages={messages}
            onSendMessage={handleSendMessage}
            isSending={isSending}
            mode={chatMode}
            modeLocked={modeLocked}
            onModeChange={handleModeChange}
            onModeReset={handleModeReset}
            modeSuggestion={modeSuggestion}
            onSuggestionAccept={handleSuggestionAccept}
            onSuggestionDismiss={handleSuggestionDismiss}
            suggestions={CHAT_SUGGESTIONS}
          />
        </section>
        <section className="dashboard__column dashboard__column--right dashboard__column--stacked-right">
          <RatingPanel scores={scores} showAccuracy />
          <AnalyticsPanel recommendedSources={summaryData?.recommended_sources ?? []} />
        </section>
      </main>
    </div>
  );
};

export default ReportDashboard;
