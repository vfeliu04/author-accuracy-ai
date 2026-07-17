import { useEffect, useState } from "react";
import type { ClaimSummary } from "../api/client";
import { API_BASE_URL, API_KEY } from "../api/client";

type ClaimsWorkspaceProps = {
  claims: ClaimSummary[];
  hasMore: boolean;
  onLoadMore: () => void;
  reportUploadId: string;
  onClose: () => void;
};

type FilterType = "ALL" | "SUPPORTED" | "CONTRADICTED" | "NOT_FOUND";

function verdictColor(verdict: string): string {
  if (verdict === "SUPPORTED") return "var(--color-success)";
  if (verdict === "CONTRADICTED") return "var(--color-danger)";
  return "var(--color-text-muted)";
}

function verdictLabel(verdict: string): string {
  if (verdict === "SUPPORTED") return "Supported";
  if (verdict === "CONTRADICTED") return "Contradicted";
  return "Not found";
}

function usePdfBlob(uploadId: string | null | undefined): string | null {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!uploadId) {
      setBlobUrl(null);
      return;
    }
    let objectUrl = "";
    fetch(`${API_BASE_URL}/api/uploads/${uploadId}/file`, {
      headers: API_KEY ? { "X-API-Key": API_KEY } : {},
    })
      .then((r) => r.blob())
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      })
      .catch(() => setBlobUrl(null));
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [uploadId]);

  return blobUrl;
}

const ClaimsWorkspace = ({
  claims,
  hasMore,
  onLoadMore,
  reportUploadId,
  onClose,
}: ClaimsWorkspaceProps) => {
  const [selectedClaim, setSelectedClaim] = useState<ClaimSummary | null>(null);
  const [filter, setFilter] = useState<FilterType>("ALL");

  const filteredClaims = filter === "ALL" ? claims : claims.filter(c => c.verdict === filter);

  // Lock body scroll while open
  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, []);

  // Auto-select first claim
  useEffect(() => {
    if (filteredClaims.length > 0 && !selectedClaim) {
      setSelectedClaim(filteredClaims[0]);
    }
  }, [filteredClaims, selectedClaim]);

  // Keyboard navigation
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      e.preventDefault();
      if (filteredClaims.length === 0) return;
      const idx = selectedClaim ? filteredClaims.findIndex(c => c.claim_id === selectedClaim.claim_id) : -1;
      if (e.key === "ArrowDown") setSelectedClaim(filteredClaims[Math.min(idx + 1, filteredClaims.length - 1)]);
      else setSelectedClaim(filteredClaims[Math.max(idx - 1, 0)]);
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [selectedClaim, filteredClaims]);

  const reportBlobUrl = usePdfBlob(reportUploadId);
  const sourceId = selectedClaim?.evidence?.[0]?.source_id ?? null;
  const sourceBlobUrl = usePdfBlob(sourceId);

  const reportPage = selectedClaim?.parent_page ?? null;
  const sourcePage = selectedClaim?.evidence?.[0]?.page ?? null;

  const reportSrc = reportBlobUrl
    ? reportPage
      ? `${reportBlobUrl}#page=${reportPage}`
      : reportBlobUrl
    : "";

  const sourceSrc = sourceBlobUrl
    ? sourcePage
      ? `${sourceBlobUrl}#page=${sourcePage}`
      : sourceBlobUrl
    : "";

  const sourceLabel = selectedClaim?.evidence?.[0]
    ? `${selectedClaim.evidence[0].source_name}${sourcePage ? ` · p.${sourcePage}` : ""}`
    : "No source";

  const reportLabel = selectedClaim
    ? `Report${reportPage ? ` · p.${reportPage}` : ""}`
    : "Report";

  return (
    <div className="claims-workspace">
      <header className="claims-workspace__header">
        <h2>Claims Workspace</h2>
        {selectedClaim && (
          <span className="claims-workspace__selected-verdict" style={{ color: verdictColor(selectedClaim.verdict) }}>
            {verdictLabel(selectedClaim.verdict)}
          </span>
        )}
        <button
          type="button"
          className="chat__fullscreen-button"
          onClick={onClose}
          aria-label="Close workspace"
        >
          ✕
        </button>
      </header>

      <div className="claims-workspace__body">
        {/* Left: claims list */}
        <div className="claims-workspace__list-col">
          <div className="claims-workspace__panel-header">
            All Claims ({filteredClaims.length}{hasMore ? "+" : ""})
          </div>
          <div className="claims-workspace__filter-bar">
            {(["ALL", "SUPPORTED", "CONTRADICTED", "NOT_FOUND"] as FilterType[]).map((f) => (
              <button
                key={f}
                type="button"
                className={`claims-workspace__filter-btn${filter === f ? " claims-workspace__filter-btn--active" : ""}`}
                onClick={() => setFilter(f)}
              >
                {f === "ALL" ? "All" : f === "SUPPORTED" ? "Supported" : f === "CONTRADICTED" ? "Contradicted" : "Not Found"}
              </button>
            ))}
          </div>
          <div className="claims-workspace__list">
            {filteredClaims.map((claim) => (
              <button
                key={claim.claim_id}
                type="button"
                className={`claims-workspace__row${selectedClaim?.claim_id === claim.claim_id ? " claims-workspace__row--active" : ""}`}
                onClick={() => setSelectedClaim(claim)}
              >
                <span
                  className="claims__verdict-dot"
                  style={{ background: verdictColor(claim.verdict), flexShrink: 0 }}
                />
                <span className="claims-workspace__claim-text">{claim.text}</span>
              </button>
            ))}
            {hasMore && (
              <button type="button" className="claims__load-more" onClick={onLoadMore}>
                Load more
              </button>
            )}
          </div>
        </div>

        {/* Middle: report PDF */}
        <div className="claims-workspace__pdf-col">
          <div className="claims-workspace__panel-header">{reportLabel}</div>
          {reportSrc ? (
            <iframe
              key={`report-${selectedClaim?.claim_id}-${reportPage}`}
              src={reportSrc}
              className="claims-workspace__iframe"
              title="Report PDF"
            />
          ) : (
            <div className="claims-workspace__empty">
              {reportBlobUrl ? "Select a claim" : "Loading report…"}
            </div>
          )}
          {selectedClaim && (
            <div className="claims-workspace__context-box claims-workspace__context-box--claim">
              <span className="claims-workspace__context-label">Claim</span>
              <p className="claims-workspace__context-text">{selectedClaim.text}</p>
            </div>
          )}
        </div>

        {/* Right: source PDF */}
        <div className="claims-workspace__pdf-col">
          <div className="claims-workspace__panel-header">{sourceLabel}</div>
          {selectedClaim?.evidence?.[0] ? (
            sourceSrc ? (
              <iframe
                key={`source-${selectedClaim?.claim_id}-${sourcePage}`}
                src={sourceSrc}
                className="claims-workspace__iframe"
                title="Source PDF"
              />
            ) : (
              <div className="claims-workspace__empty">Loading source…</div>
            )
          ) : (
            <div className="claims-workspace__empty">
              {selectedClaim ? "No source evidence for this claim" : "Select a claim"}
            </div>
          )}
          {selectedClaim?.evidence?.[0]?.snippet && (
            <div className="claims-workspace__context-box claims-workspace__context-box--evidence">
              <span className="claims-workspace__context-label">Evidence snippet</span>
              <blockquote className="claims-workspace__context-text claims-workspace__context-text--quote">
                "{selectedClaim.evidence[0].snippet}"
              </blockquote>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ClaimsWorkspace;
