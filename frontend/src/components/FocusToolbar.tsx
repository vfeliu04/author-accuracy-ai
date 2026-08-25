import type { ReactNode } from "react";
import { useSearchParams } from "react-router-dom";

// The shared focus-mode toolbar: one Close behavior (drop every focus param;
// the browser Back button does the equivalent) plus whatever controls the
// mode adds.
export default function FocusToolbar({ children }: { children?: ReactNode }) {
  const [, setSearchParams] = useSearchParams();
  return (
    <div className="claims-toolbar">
      <button
        type="button"
        className="btn btn--ghost btn--small"
        onClick={() => setSearchParams({})}
      >
        ✕ Close
      </button>
      {children}
    </div>
  );
}
