import type { ReactNode } from "react";
import { Link } from "react-router-dom";

// The persistent chrome every page composes itself into: brand on the left,
// a contextual title block in the middle, page actions on the right.
export default function AppShell({
  title,
  actions,
  children
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand" aria-label="Author Accuracy home">
          <span className="brand__mark" aria-hidden>
            ✓
          </span>
          {title ? null : <h1 className="brand__name">Author Accuracy</h1>}
        </Link>
        <div className="topbar__title">{title}</div>
        <div className="topbar__actions">{actions}</div>
      </header>
      <div className="shell__body">{children}</div>
    </div>
  );
}
