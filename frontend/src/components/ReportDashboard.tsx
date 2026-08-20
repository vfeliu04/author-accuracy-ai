import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import ChatPanel, { type ChatMessage } from "./ChatPanel";
import RatingPanel from "./RatingPanel";
import { useChat, useReport, useRetryRun, useRun } from "../api/queries";
import type { ChatMode, Claim, Verdict } from "../api/types";

const PIPELINE_STEPS = ["ingest", "extract", "verify", "score"];

// Pending steps have no server-sent entry yet; without this fallback the feed
// showed raw internal names ("score") below the labeled completed steps.
const STEP_FALLBACK_LABELS: Record<string, string> = {
  ingest: "Ingest documents",
  extract: "Extract claims",
  verify: "Verify claims against sources",
  score: "Score the report"
};

// Raw exception text is for logs; users get the translation (the raw message
// stays visible in a collapsible block below).
const ERROR_HINTS: Array<{ match: RegExp; hint: string }> = [
  {
    match: /APIConnectionError|Connection error/i,
    hint: "The server lost its network connection mid-run (laptop sleep or dropped Wi-Fi are the usual causes). Retrying resumes where it stopped."
  },
  {
    match: /TimeoutError.*Batch|still 'in_progress'/i,
    hint: "The verification batch was still queued on the provider's side when the app stopped waiting. The batch keeps its place — retrying reattaches to it at no extra cost."
  },
  {
    match: /credit balance|billing/i,
    hint: "The API account looks out of credit — top up at console.anthropic.com, then retry."
  }
];

function humanizeError(error: string | null): string | null {
  if (!error) return null;
  return ERROR_HINTS.find(({ match }) => match.test(error))?.hint ?? null;
}

function elapsedLabel(sinceIso: string | undefined): string | null {
  if (!sinceIso) return null;
  const started = new Date(sinceIso).getTime();
  if (Number.isNaN(started)) return null;
  const minutes = Math.max(0, Math.round((Date.now() - started) / 60000));
  return minutes < 1 ? "just started" : minutes < 60 ? `running for ${minutes} min` : `running for ${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

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
            <span
              className={`pipeline-progress__icon${status === "running" ? " pipeline-progress__icon--spinning" : ""}`}
            >
              {icon}
            </span>
            <span>{entry?.label ?? STEP_FALLBACK_LABELS[step] ?? step}</span>
            {step === "verify" && status === "running" ? (
              <span className="pipeline-progress__hint">
                Verification runs through the half-price batch API — usually minutes, but it can
                queue for longer. This page keeps itself updated.
              </span>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

export function ClaimBadges({ claim }: { claim: Claim }) {
  return (
    <span className="claim-row__badges">
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
      {claim.downgraded ? (
        <span className="source-pill__badge" title="The quote check failed; the verdict was downgraded">
          downgraded
        </span>
      ) : null}
    </span>
  );
}

function ClaimRow({ claim }: { claim: Claim }) {
  const evidence = claim.evidence_source;
  return (
    <div className="claim-row">
      <ClaimBadges claim={claim} />
      <span className="claim-row__text">{claim.text}</span>
      <p className="claim-row__rationale">{claim.rationale}</p>
      {claim.quote && evidence ? (
        <p className="claim-row__quote">
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
  const retry = useRetryRun(runId);

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
                <span className="dashboard__subtitle">{elapsedLabel(job?.created_at)}</span>
              </header>
              {job ? <ProgressFeed progress={job.progress} /> : <p className="dashboard__status">Queued…</p>}
            </article>
          </section>
        </main>
      ) : null}

      {status === "FAILED" ? (
        <main className="dashboard__content">
          <section className="dashboard__column" style={{ width: "100%", gridColumn: "1 / -1" }}>
            <article className="card">
              <header className="card__header">
                <h2>This run failed</h2>
              </header>
              {humanizeError(run?.error ?? null) ? (
                <p className="card__body-text">{humanizeError(run?.error ?? null)}</p>
              ) : null}
              <details style={{ marginBottom: "0.8rem" }}>
                <summary className="dashboard__status dashboard__status--error" style={{ cursor: "pointer" }}>
                  Technical detail
                </summary>
                <p className="dashboard__status">{run?.error ?? "check the server logs"}</p>
              </details>
              {job ? <ProgressFeed progress={job.progress} /> : null}
              <div style={{ marginTop: "0.8rem" }}>
                <button
                  type="button"
                  className="dashboard__refresh-button"
                  disabled={retry.isPending}
                  onClick={() => retry.mutate()}
                >
                  {retry.isPending ? "Retrying…" : "Retry run"}
                </button>
                <p style={{ margin: "0.4rem 0 0", fontSize: "0.85em", opacity: 0.75 }}>
                  Picks up from the first incomplete step — documents that were already
                  ingested are kept.
                </p>
                {retry.error ? (
                  <p className="dashboard__status dashboard__status--error">
                    Retry failed: {retry.error instanceof Error ? retry.error.message : "unknown error"}
                  </p>
                ) : null}
              </div>
            </article>
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
                    <span className="upload__item-name" title={source.title ?? source.doc_id}>
                      {source.title ?? source.doc_id.slice(0, 8)}
                    </span>
                    <span className="source-pill__badge">{source.tier}</span>
                    <span className="source-pill__badge" title="Credibility score">
                      {Math.round(source.total)}/100
                    </span>
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
