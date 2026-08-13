import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useRuns } from "../api/queries";
import type { RunStatus } from "../api/types";

const STATUS_LABEL: Record<RunStatus, string> = {
  CREATED: "Queued",
  RUNNING: "Running",
  DONE: "Complete",
  FAILED: "Failed"
};

const STATUS_CLASS: Record<RunStatus, string> = {
  CREATED: "dashboard__badge dashboard__badge--queued",
  RUNNING: "dashboard__badge dashboard__badge--running",
  DONE: "dashboard__badge dashboard__badge--done",
  FAILED: "dashboard__badge dashboard__badge--failed"
};

function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

// Every run is retained (v2 never resets), so this lists them all. There is no
// per-report title in v2, so a run is identified by its time, status, and a
// short id.
const HistoryPage = () => {
  const navigate = useNavigate();
  const { data: runs, isLoading, error } = useRuns();
  // Pick two runs to diff — keep the two most recent selections.
  const [selected, setSelected] = useState<string[]>([]);
  const toggleCompare = (id: string) =>
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(-2)
    );

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <div className="dashboard__title-block">
          <h1>Run History</h1>
          <p className="dashboard__subtitle">Every fact-check run, newest first.</p>
        </div>
        <div className="dashboard__meta">
          {selected.length === 2 ? (
            <button
              type="button"
              className="dashboard__refresh-button"
              onClick={() => navigate(`/compare?a=${selected[0]}&b=${selected[1]}`)}
            >
              Compare selected →
            </button>
          ) : null}
          <button type="button" className="dashboard__refresh-button" onClick={() => navigate("/")}>
            + New run
          </button>
        </div>
      </header>
      <main className="dashboard__content">
        <section className="dashboard__column" style={{ width: "100%", gridColumn: "1 / -1" }}>
          {isLoading ? <p className="dashboard__status">Loading runs…</p> : null}
          {error ? (
            <p className="dashboard__status dashboard__status--error">
              {error instanceof Error ? error.message : "Failed to load runs."}
            </p>
          ) : null}
          {runs && runs.length === 0 ? (
            <p className="dashboard__status">No runs yet — create one from the upload page.</p>
          ) : null}
          <div className="upload__list">
            {runs?.map((run) => (
              <div key={run.id} className="upload__item upload__item--with-actions">
                <Link
                  to={`/runs/${run.id}`}
                  style={{ display: "flex", gap: 8, alignItems: "center", flex: 1, textDecoration: "none" }}
                >
                  <span className="upload__item-name">{formatTimestamp(run.created_at)}</span>
                  <span className="source-pill__badge">{run.id.slice(0, 8)}</span>
                  <span className={STATUS_CLASS[run.status]}>{STATUS_LABEL[run.status]}</span>
                </Link>
                <button
                  type="button"
                  className={selected.includes(run.id) ? "pill pill--outlined" : "pill pill--ghost"}
                  onClick={() => toggleCompare(run.id)}
                >
                  {selected.includes(run.id) ? "✓ compare" : "compare"}
                </button>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
};

export default HistoryPage;
