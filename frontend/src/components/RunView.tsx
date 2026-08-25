import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useChat, useReport, useRetryRun, useRun } from "../api/queries";
import type { ChatMode } from "../api/types";
import { humanizeError } from "../lib/errors";
import AnalysisPanel from "./AnalysisPanel";
import AppShell from "./AppShell";
import ChatPanel, { type ChatMessage } from "./ChatPanel";
import ProgressFeed from "./ProgressFeed";
import SourcesPanel from "./SourcesPanel";
import StatusChip from "./StatusChip";

const CHAT_SUGGESTIONS = [
  "Which claims are contradicted?",
  "What is the weakest supported claim?",
  "Which source backs the most claims?"
];

// The run page: Sources | Chat (or progress) | Analysis, switching its
// center on run status. Chat history lives here so it survives everything
// short of leaving the run.
export default function RunView() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const runQuery = useRun(runId);
  const reportQuery = useReport(runId);
  const chat = useChat(runId ?? "");
  const retry = useRetryRun(runId);

  const [mode, setMode] = useState<ChatMode>("evidence");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatError, setChatError] = useState<string | null>(null);

  const run = runQuery.data?.run;
  const job = runQuery.data?.job;
  const uploads = runQuery.data?.uploads ?? [];
  const report = reportQuery.data;
  const status = run?.status;

  const handleSend = (text: string) => {
    setChatError(null);
    const history = messages.map((message) => ({
      role: message.author,
      content: message.text
    }));
    setMessages((prev) => [...prev, { id: Date.now(), author: "user", text }]);
    chat.mutate(
      { question: text, history, mode },
      {
        onSuccess: (data) =>
          setMessages((prev) => [
            ...prev,
            { id: Date.now() + 1, author: "assistant", text: data.answer }
          ]),
        // Surface the failure separately — do NOT push it into `messages`, or
        // the next turn would send a fabricated assistant reply as history.
        onError: (err) =>
          setChatError(err instanceof Error ? err.message : "The chat service is unavailable.")
      }
    );
  };

  const title = run?.title ?? (runId ? `Run ${runId.slice(0, 8)}` : "Run");

  const openSource = () => navigate(`/runs/${runId}/sources/${report?.sources[0]?.doc_id ?? ""}`);

  return (
    <AppShell
      title={
        <>
          <button
            type="button"
            className="back-btn"
            onClick={() => navigate("/")}
            aria-label="Back to all runs"
          >
            ←
          </button>
          <h1>{title}</h1>
          {status ? <StatusChip status={status} /> : null}
        </>
      }
    >
      {runQuery.error ? (
        <div className="panels">
          <main className="panel panel--main">
            <div className="panel__body">
              <p className="error-text">
                {runQuery.error instanceof Error
                  ? runQuery.error.message
                  : "Failed to load this run."}
              </p>
            </div>
          </main>
        </div>
      ) : status === "DONE" && report ? (
        <div className="panels">
          <SourcesPanel
            uploads={uploads}
            report={report}
            onOpenSource={(docId) => navigate(`/runs/${runId}/sources/${docId}`)}
          />
          <ChatPanel
            messages={messages}
            onSendMessage={handleSend}
            isSending={chat.isPending}
            mode={mode}
            onModeChange={setMode}
            suggestions={CHAT_SUGGESTIONS}
            error={chatError}
          />
          <AnalysisPanel
            report={report}
            onOpenClaims={() => navigate(`/runs/${runId}/workspace`)}
            onOpenReport={() => navigate(`/runs/${runId}/report`)}
            onOpenCredibility={openSource}
          />
        </div>
      ) : status === "FAILED" ? (
        <div className="panels">
          <SourcesPanel uploads={uploads} report={report} />
          <main className="panel panel--main">
            <div className="panel__body">
              <div className="failure-card">
                <h3>This verification failed</h3>
                {humanizeError(run?.error ?? null) ? (
                  <p style={{ margin: 0 }}>{humanizeError(run?.error ?? null)}</p>
                ) : null}
                <details>
                  <summary>Technical detail</summary>
                  <p className="error-text">{run?.error ?? "check the server logs"}</p>
                </details>
                <div>
                  <button
                    type="button"
                    className="btn btn--primary"
                    disabled={retry.isPending}
                    onClick={() => retry.mutate()}
                  >
                    {retry.isPending ? "Retrying…" : "Retry run"}
                  </button>
                  <p className="muted" style={{ margin: "0.4rem 0 0", fontSize: "0.82rem" }}>
                    Picks up from the first incomplete step — documents already read are kept.
                  </p>
                  {retry.error ? (
                    <p className="error-text">
                      Retry failed:{" "}
                      {retry.error instanceof Error ? retry.error.message : "unknown error"}
                    </p>
                  ) : null}
                </div>
                {job ? <ProgressFeed progress={job.progress} heading={false} note={false} /> : null}
              </div>
            </div>
          </main>
          <AnalysisPanel report={report} />
        </div>
      ) : (
        <div className="panels">
          <SourcesPanel uploads={uploads} report={report} />
          <main className="panel panel--main">
            <div className="panel__head">
              <h2>Progress</h2>
            </div>
            <div className="panel__body">
              {job ? (
                <ProgressFeed progress={job.progress} />
              ) : (
                <p className="muted">{status ? "Queued…" : "Loading…"}</p>
              )}
            </div>
            <div className="chat-bottom">
              <div className="chat-input chat-input--disabled">
                <input placeholder="Chat unlocks when verification completes…" disabled />
                <button type="button" className="chat-input__send" disabled aria-hidden>
                  ➤
                </button>
              </div>
            </div>
          </main>
          <AnalysisPanel report={report} />
        </div>
      )}
    </AppShell>
  );
}
