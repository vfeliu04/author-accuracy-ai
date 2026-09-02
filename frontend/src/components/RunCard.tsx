import { useEffect, useRef, useState } from "react";
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
  onOpen,
  onDelete
}: {
  run: RunListItem;
  picked?: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const retry = useRetryRun(run.id);
  const title = run.title ?? `Run ${run.id.slice(0, 8)}`;
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (event: globalThis.MouseEvent) => {
      if (menuRef.current && event.target instanceof Node && !menuRef.current.contains(event.target)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuOpen]);

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
        // Only the card itself — bubbled keydowns from the kebab/menu/retry
        // buttons must keep their native activation, not navigate.
        if (event.target !== event.currentTarget) return;
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
        <div className="card__menu-holder" ref={menuRef}>
          <button
            type="button"
            className="card__kebab"
            aria-label="Run options"
            onClick={(event) => {
              event.stopPropagation();
              setMenuOpen((open) => !open);
            }}
          >
            ⋮
          </button>
          {menuOpen ? (
            <div className="card-menu">
              <button
                type="button"
                className="card-menu__item card-menu__item--danger"
                onClick={(event) => {
                  event.stopPropagation();
                  setMenuOpen(false);
                  onDelete();
                }}
              >
                Delete verification…
              </button>
            </div>
          ) : null}
        </div>
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
