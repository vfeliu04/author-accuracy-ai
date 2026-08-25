import type { MouseEvent } from "react";
import { useRetryRun } from "../api/queries";
import type { RunListItem } from "../api/types";
import { emojiFor } from "../lib/emoji";
import { formatDate } from "../lib/format";
import StatusChip from "./StatusChip";

function ScorePill({
  kind,
  label,
  value
}: {
  kind: "a" | "c" | "v";
  label: string;
  value: number | null;
}) {
  if (value === null) return null;
  return (
    <span className={`score-pill score-pill--${kind}`}>
      {label} {Math.round(value * 100)}
    </span>
  );
}

export default function RunCard({
  run,
  picked = false,
  onOpen
}: {
  run: RunListItem;
  picked?: boolean;
  onOpen: () => void;
}) {
  const retry = useRetryRun(run.id);
  const title = run.title ?? `Run ${run.id.slice(0, 8)}`;

  const handleRetry = (event: MouseEvent) => {
    event.stopPropagation();
    retry.mutate();
  };

  return (
    <div
      className={`card${picked ? " card--picked" : ""}`}
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen();
        }
      }}
    >
      <div className="card__top">
        <span className="card__emoji" aria-hidden>
          {emojiFor(run.title ?? run.id)}
        </span>
      </div>
      <div className="card__title">{title}</div>
      <div className="card__meta">
        {formatDate(run.created_at)}
        {run.source_count !== null
          ? ` · ${run.source_count} source${run.source_count === 1 ? "" : "s"}`
          : ""}
      </div>
      <div className="card__foot">
        {run.status === "DONE" && run.scores ? (
          <>
            <ScorePill kind="a" label="A" value={run.scores.accuracy} />
            <ScorePill kind="c" label="C" value={run.scores.credibility} />
            <ScorePill kind="v" label="V" value={run.scores.validity} />
          </>
        ) : run.status === "FAILED" ? (
          <>
            <StatusChip status="FAILED" />
            <button
              type="button"
              className="btn btn--ghost btn--small"
              disabled={retry.isPending}
              onClick={handleRetry}
            >
              {retry.isPending ? "Retrying…" : "↻ Retry"}
            </button>
            {retry.error ? (
              <span className="error-text">
                {retry.error instanceof Error ? retry.error.message : "Retry failed"}
              </span>
            ) : null}
          </>
        ) : (
          <StatusChip status={run.status} />
        )}
      </div>
    </div>
  );
}
