import type { RunStatus } from "../api/types";

const LABELS: Record<RunStatus, string> = {
  CREATED: "Queued",
  RUNNING: "Running",
  DONE: "Done",
  FAILED: "Failed"
};

const CLASSES: Record<RunStatus, string> = {
  CREATED: "chip--running",
  RUNNING: "chip--running",
  DONE: "chip--done",
  FAILED: "chip--failed"
};

export default function StatusChip({ status, label }: { status: RunStatus; label?: string }) {
  const moving = status === "RUNNING" || status === "CREATED";
  return (
    <span className={`chip ${CLASSES[status]}`}>
      {moving ? (
        <span className="spinner" aria-hidden>
          ⟳
        </span>
      ) : null}
      {label ?? LABELS[status]}
    </span>
  );
}
