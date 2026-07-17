import { useState } from "react";
import type { ClaimSummary } from "../api/client";

type ClaimsPanelProps = {
  claims: ClaimSummary[];
  totalClaims: number;
  hasMore: boolean;
  onLoadMore: () => void;
  onExpand: () => void;
};

function verdictColor(verdict: string): string {
  if (verdict === "SUPPORTED") return "var(--color-success)";
  if (verdict === "CONTRADICTED") return "var(--color-danger)";
  return "var(--color-text-muted)";
}

const ClaimsPanel = ({ claims, totalClaims, hasMore, onLoadMore, onExpand }: ClaimsPanelProps) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <article className="card">
      <header className="card__header">
        <h2>Claims</h2>
        <span className="upload__count">{totalClaims}</span>
        <button
          type="button"
          className="chat__fullscreen-button"
          onClick={onExpand}
          aria-label="Expand claims workspace"
        >
          ⛶
        </button>
      </header>
      <div className="claims__list">
        {claims.map((claim) => (
          <div key={claim.claim_id} className="claims__item">
            <button
              type="button"
              className="claims__row"
              onClick={() => setExpandedId(expandedId === claim.claim_id ? null : claim.claim_id)}
            >
              <span className="claims__verdict-dot" style={{ background: verdictColor(claim.verdict) }} />
              <span className="claims__text">{claim.text}</span>
              <span className="claims__band">{claim.confidence_band}</span>
            </button>
            {expandedId === claim.claim_id && (
              <div className="claims__detail">
                <p className="claims__explanation">{claim.explanation}</p>
                {claim.evidence?.map((ev, i) => (
                  <div key={i} className="claims__evidence">
                    <span className="claims__evidence-source">
                      {ev.source_name}{ev.page ? ` · p.${ev.page}` : ""}
                    </span>
                    <blockquote className="claims__snippet">"{ev.snippet}"</blockquote>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      {hasMore && (
        <button type="button" className="claims__load-more" onClick={onLoadMore}>
          Load more
        </button>
      )}
    </article>
  );
};

export default ClaimsPanel;
