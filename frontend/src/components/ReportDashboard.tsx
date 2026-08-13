import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import ChatPanel, { type ChatMessage } from "./ChatPanel";
import RatingPanel from "./RatingPanel";
import { useChat, useReport, useRun } from "../api/queries";
import type { ChatMode, Claim, Verdict } from "../api/types";

const PIPELINE_STEPS = ["ingest", "extract", "verify", "score"];

const CHAT_SUGGESTIONS = [
  "Which claims are contradicted?",
  "What is the weakest supported claim?",
  "How can I improve the report's accuracy?",
  "Which source backs the most claims?"
];

const VERDICT_COLOR: Record<Verdict, string> = {
  SUPPORTED: "var(--color-success)",
  CONTRADICTED: "var(--color-danger)",
  UNVERIFIABLE: "var(--color-warning)"
};

function pct(fraction: number | null): string {
  return fraction === null ? "—" : `${Math.round(fraction * 100)}%`;
}

function ProgressFeed({ progress }: { progress: { step: string; label: string; status: string }[] }) {
  const byStep = Object.fromEntries(progress.map((entry) => [entry.step, entry]));
  return (
    <ul className="pipeline-progress">
      {PIPELINE_STEPS.map((step) => {
        const entry = byStep[step];
        const status = entry?.status ?? "pending";
        const icon = status === "done" ? "✓" : status === "running" ? "⟳" : status === "failed" ? "✗" : "·";
        return (
          <li key={step} className={`pipeline-progress__step pipeline-progress__step--${status}`}>
            <span className="pipeline-progress__icon">{icon}</span>
            <span>{entry?.label ?? step}</span>
          </li>
        );
      })}
    </ul>
  );
}

function ClaimRow({ claim }: { claim: Claim }) {
  const evidence = claim.evidence_source;
  return (
    <div className="upload__item" style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}>
      {/* flex-wrap + a min basis on the text: in the narrow claims column the
          badges would otherwise squeeze the text to a word-per-line sliver. */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%", flexWrap: "wrap" }}>
        <span
          className="source-pill__badge"
          style={{ background: VERDICT_COLOR[claim.verdict], color: "#fff" }}
        >
          {claim.verdict}
        </span>
        {claim.stance === "disavowed" ? (
          <span className="source-pill__badge" title="The report itself marks this claim false">
            disavowed
          </span>
        ) : null}
        {claim.downgraded ? <span className="source-pill__badge">downgraded</span> : null}
        <span className="upload__item-name" style={{ whiteSpace: "normal", flex: "1 1 200px" }}>
          {claim.text}
        </span>
      </div>
      <p style={{ margin: 0, fontSize: "0.85em", opacity: 0.8 }}>{claim.rationale}</p>
      {claim.quote && evidence ? (
        <p style={{ margin: 0, fontSize: "0.8em", opacity: 0.7 }}>
          “{claim.quote}” — {evidence.title ?? "source"}
          {evidence.page !== null ? ` p.${evidence.page}` : ""}
        </p>
      ) : null}
    </div>
  );
}

const ReportDashboard = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const runQuery = useRun(runId);
  const reportQuery = useReport(runId);
  const chat = useChat(runId ?? "");

  const [mode, setMode] = useState<ChatMode>("evidence");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatError, setChatError] = useState<string | null>(null);

  const run = runQuery.data?.run;
  const job = runQuery.data?.job;
  const report = reportQuery.data;
  const status = run?.status;
  const done = status === "DONE";

  const handleSend = (text: string) => {
    setChatError(null);
    const history = messages.map((m) => ({ role: m.author, content: m.text }));
    setMessages((prev) => [...prev, { id: Date.now(), author: "user", text }]);
    chat.mutate(
      { question: text, history, mode },
      {
        onSuccess: (data) =>
          setMessages((prev) => [...prev, { id: Date.now() + 1, author: "assistant", text: data.answer }]),
        // Surface the failure separately — do NOT push it into `messages`, or
        // the next turn would send a fabricated assistant reply as history.
        onError: (err) =>
          setChatError(err instanceof Error ? err.message : "The chat service is unavailable.")
      }
    );
  };

  const stats = report?.stats;

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>Run {runId?.slice(0, 8)}</h1>
          {done && stats ? (
            <p className="dashboard__subtitle">
              {stats.claims_supported} of {stats.claims_total} claims supported
              &nbsp;·&nbsp;{report?.sources.length} source{report?.sources.length !== 1 ? "s" : ""}
              &nbsp;·&nbsp;{pct(report?.scores?.accuracy ?? null)} accuracy
            </p>
          ) : (
            <p className="dashboard__subtitle">Status: {status ?? "loading"}</p>
          )}
          {runQuery.error ? (
            <p className="dashboard__status dashboard__status--error">
              {runQuery.error instanceof Error ? runQuery.error.message : "Failed to load run."}
            </p>
          ) : null}
        </div>
        <div className="dashboard__meta">
          {done ? (
            <>
              <button
                type="button"
                className="dashboard__refresh-button"
                onClick={() => navigate(`/runs/${runId}/workspace`)}
              >
                Claims workspace
              </button>
              <button
                type="button"
                className="dashboard__refresh-button"
                onClick={() => navigate(`/runs/${runId}/report`)}
              >
                Report PDF
              </button>
            </>
          ) : null}
          <button type="button" className="dashboard__refresh-button" onClick={() => navigate("/runs")}>
            ← All runs
          </button>
        </div>
      </header>

      {status && status !== "DONE" && status !== "FAILED" ? (
        <main className="dashboard__content">
          <section className="dashboard__column" style={{ width: "100%", gridColumn: "1 / -1" }}>
            <article className="card">
              <header className="card__header">
                <h2>Running the pipeline</h2>
              </header>
              {job ? <ProgressFeed progress={job.progress} /> : <p className="dashboard__status">Queued…</p>}
            </article>
          </section>
        </main>
      ) : null}

      {status === "FAILED" ? (
        <main className="dashboard__content">
          <section className="dashboard__column" style={{ width: "100%", gridColumn: "1 / -1" }}>
            <p className="dashboard__status dashboard__status--error">
              This run failed: {run?.error ?? "check the server logs"}
            </p>
          </section>
        </main>
      ) : null}

      {done && report ? (
        <main className="dashboard__content">
          <section className="dashboard__column dashboard__column--left dashboard__column--stacked-left">
            <article className="card">
              <header className="card__header">
                <h2>Sources</h2>
              </header>
              <div className="upload__list">
                {report.sources.map((source) => (
                  <button
                    key={source.doc_id}
                    type="button"
                    className="upload__item upload__item--with-actions"
                    onClick={() => navigate(`/runs/${runId}/sources/${source.doc_id}`)}
                  >
                    <span className="upload__item-name">{source.title ?? source.doc_id.slice(0, 8)}</span>
                    <span className="source-pill__badge">{source.tier}</span>
                    <span className="source-pill__badge">{Math.round(source.total)}</span>
                  </button>
                ))}
              </div>
            </article>
            <article className="card">
              <header className="card__header">
                <h2>Claims ({stats?.claims_total ?? 0})</h2>
              </header>
              <div className="upload__list">
                {report.claims.map((claim) => (
                  <ClaimRow key={claim.claim_id} claim={claim} />
                ))}
              </div>
            </article>
          </section>
          <section className="dashboard__column dashboard__column--center">
            <ChatPanel
              messages={messages}
              onSendMessage={handleSend}
              isSending={chat.isPending}
              mode={mode}
              onModeChange={setMode}
              suggestions={CHAT_SUGGESTIONS}
            />
            {chatError ? (
              <p className="dashboard__status dashboard__status--error">
                The chat service is unavailable: {chatError}
              </p>
            ) : null}
          </section>
          <section className="dashboard__column dashboard__column--right dashboard__column--stacked-right">
            <RatingPanel scores={report.scores} />
          </section>
        </main>
      ) : null}
    </div>
  );
};

export default ReportDashboard;
