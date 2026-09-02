import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDeleteRun, useRuns } from "../api/queries";
import type { RunListItem } from "../api/types";
import AppShell from "./AppShell";
import RunCard from "./RunCard";
import UploadDialog from "./UploadDialog";

type Filter = "ALL" | "DONE" | "RUNNING" | "FAILED";

const FILTERS: Array<{ key: Filter; label: string }> = [
  { key: "ALL", label: "All" },
  { key: "DONE", label: "Done" },
  { key: "RUNNING", label: "Running" },
  { key: "FAILED", label: "Failed" }
];

function matches(run: RunListItem, filter: Filter): boolean {
  if (filter === "ALL") return true;
  if (filter === "RUNNING") return run.status === "RUNNING" || run.status === "CREATED";
  return run.status === filter;
}

export default function HomePage() {
  const navigate = useNavigate();
  const runsQuery = useRuns();
  const [filter, setFilter] = useState<Filter>("ALL");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [pickMode, setPickMode] = useState(false);
  const [picks, setPicks] = useState<string[]>([]);
  const [deleteTarget, setDeleteTarget] = useState<RunListItem | null>(null);
  const deletion = useDeleteRun();

  const runs = runsQuery.data ?? [];
  const visible = runs.filter((run) => matches(run, filter));

  const openCard = (run: RunListItem) => {
    if (!pickMode) {
      navigate(`/runs/${run.id}`);
      return;
    }
    const next = picks.includes(run.id)
      ? picks.filter((id) => id !== run.id)
      : [...picks, run.id];
    if (next.length === 2) {
      navigate(`/compare?a=${next[0]}&b=${next[1]}`);
      return;
    }
    setPicks(next);
  };

  const togglePickMode = () => {
    setPickMode((prev) => !prev);
    setPicks([]);
  };

  return (
    <AppShell
      actions={
        <>
          <button type="button" className="btn btn--ghost" onClick={togglePickMode}>
            {pickMode ? (
              "✕ Cancel compare"
            ) : (
              <>
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden
                >
                  <path d="M8 3 4 7l4 4" />
                  <path d="M4 7h16" />
                  <path d="m16 21 4-4-4-4" />
                  <path d="M20 17H4" />
                </svg>
                Compare runs
              </>
            )}
          </button>
          <button type="button" className="btn btn--primary" onClick={() => setDialogOpen(true)}>
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              aria-hidden
            >
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New verification
          </button>
        </>
      }
    >
      <div className="home-body">
        <div className="home-inner">
          <div className="home-controls">
            {FILTERS.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                className={`filter-chip${filter === key ? " active" : ""}`}
                onClick={() => setFilter(key)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="section-title">
            <h2>Recent verifications</h2>
            {pickMode ? (
              <span className="muted">Pick two runs to compare ({picks.length}/2)</span>
            ) : null}
          </div>

          {runsQuery.isLoading ? (
            <p className="muted">Loading…</p>
          ) : runsQuery.error ? (
            <p className="error-text">
              {runsQuery.error instanceof Error
                ? runsQuery.error.message
                : "Could not load your verifications."}
            </p>
          ) : (
            <div className="cards">
              <div
                className="card card--create"
                role="button"
                tabIndex={0}
                aria-label="Start a new verification"
                onClick={() => setDialogOpen(true)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setDialogOpen(true);
                  }
                }}
              >
                <div className="plus" aria-hidden>
                  ＋
                </div>
                <span>New verification</span>
              </div>
              {visible.map((run) => (
                <RunCard
                  key={run.id}
                  run={run}
                  picked={picks.includes(run.id)}
                  onOpen={() => openCard(run)}
                  onDelete={() => {
                    deletion.reset();
                    setDeleteTarget(run);
                  }}
                />
              ))}
            </div>
          )}

          {!runsQuery.isLoading && !runsQuery.error && runs.length === 0 ? (
            <p className="empty-state">
              No verifications yet — upload a report and its sources to start.
            </p>
          ) : null}
        </div>
      </div>
      {dialogOpen ? <UploadDialog onClose={() => setDialogOpen(false)} /> : null}
      {deleteTarget ? (
        <div
          className="modal-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) setDeleteTarget(null);
          }}
        >
          <div className="modal modal--confirm" role="dialog" aria-modal="true" aria-label="Delete verification">
            <div className="modal__head">
              <h2>Delete verification</h2>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setDeleteTarget(null)}
                aria-label="Close"
              >
                ✕
              </button>
            </div>
            <div className="modal__body">
              <p style={{ margin: "0.4rem 0 0.9rem" }}>
                This permanently removes{" "}
                <strong>“{deleteTarget.title ?? `Run ${deleteTarget.id.slice(0, 8)}`}”</strong> —
                its analysis, verdicts, and uploaded PDFs. It can&rsquo;t be undone.
              </p>
              {deletion.error ? (
                <p className="modal__error">
                  {deletion.error instanceof Error
                    ? deletion.error.message
                    : "The deletion failed."}
                </p>
              ) : null}
            </div>
            <div className="modal__foot">
              <span />
              <div className="modal__actions">
                <button type="button" className="btn btn--ghost" onClick={() => setDeleteTarget(null)}>
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn--danger"
                  disabled={deletion.isPending}
                  onClick={() =>
                    deletion.mutate(deleteTarget.id, {
                      onSuccess: () => setDeleteTarget(null)
                    })
                  }
                >
                  {deletion.isPending ? "Deleting…" : "Delete verification"}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </AppShell>
  );
}
