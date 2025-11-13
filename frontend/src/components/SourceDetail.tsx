import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import SummaryPanel from "./SummaryPanel";
import RatingPanel from "./RatingPanel";
import { useReportData } from "../context/ReportDataContext";
import { getSourceDetail, type SourceDetailResponse, API_BASE_URL } from "../api/client";

type SourceDetailWithValidity = SourceDetailResponse & {
  validity?: {
    score: number;
    supported?: number;
    total?: number;
  } | null;
};

const fallbackScores = { overall: 0, accuracy: 0, credibility: 0, validity: 0 };

// SourceDetail displays a selected internal source alongside supporting context.
const SourceDetail = () => {
  const navigate = useNavigate();
  const { sourceId } = useParams();
  const { getInternalSourceById, summaryData } = useReportData();
  const [detail, setDetail] = useState<SourceDetailWithValidity | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [claimPage, setClaimPage] = useState(0);
  const pageSize = 5;

  useEffect(() => {
    let isMounted = true;
    async function fetchDetail() {
      if (!sourceId) {
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const response = await getSourceDetail(sourceId, claimPage, pageSize);
        if (isMounted) {
          setDetail((prev) => {
            const mergedClaims =
              claimPage === 0 ? response.claims : [...(prev?.claims ?? []), ...response.claims];
            return {
              ...response,
              claims: mergedClaims
            };
          });
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Unable to load source details.");
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }
    fetchDetail();
    return () => {
      isMounted = false;
    };
  }, [sourceId, claimPage, pageSize]);

  const evaluatedSource = summaryData?.sources.find((source) => source.id === sourceId);
  const uploadedSource = sourceId ? getInternalSourceById(sourceId) : undefined;
  const displayName = detail?.upload.file_name ?? evaluatedSource?.name ?? uploadedSource?.name;
  const filePath =
    detail?.upload.file_url ?? evaluatedSource?.file_url ?? uploadedSource?.filePath;

  useEffect(() => {
    setClaimPage(0);
  }, [sourceId]);

  if (loading) {
    return (
      <div className="source-detail source-detail--missing">
        <div className="source-detail__missing-card">
          <p>Loading source information…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="source-detail source-detail--missing">
        <div className="source-detail__missing-card">
          <p>{error}</p>
          <button type="button" className="pill pill--filled" onClick={() => navigate(-1)}>
            Return to dashboard
          </button>
        </div>
      </div>
    );
  }

  if (!displayName || !filePath) {
    return (
      <div className="source-detail source-detail--missing">
        <div className="source-detail__missing-card">
          <p>We couldn&apos;t find that source.</p>
          <button type="button" className="pill pill--filled" onClick={() => navigate("/")}>
            Return to dashboard
          </button>
        </div>
      </div>
    );
  }

  const credibilityFromDetail = detail?.credibility?.score
    ? Math.min(Math.max(detail.credibility.score / 100, 0), 1)
    : null;

  const detailValidityScore =
    detail && (detail as any).validity && typeof (detail as any).validity.score === "number"
      ? (detail as any).validity.score
      : null;

  const validityScore = detailValidityScore ?? uploadedSource?.scores?.validity ?? null;

  const availableScores = [
    credibilityFromDetail,
    typeof validityScore === "number" ? validityScore : null
  ].filter((value): value is number => value !== null && !Number.isNaN(value));

  const ratingScores =
    (availableScores.length
      ? {
          overall:
            availableScores.reduce((total, value) => total + value, 0) / availableScores.length,
          credibility: credibilityFromDetail ?? availableScores[0],
          validity:
            detailValidityScore !== null
              ? detailValidityScore
              : availableScores[availableScores.length - 1],
          accuracy: undefined
        }
      : summaryData?.scores) ??
    (uploadedSource?.scores
      ? {
          overall:
            uploadedSource.scores.overall ??
            (uploadedSource.scores.credibility + uploadedSource.scores.validity) / 2,
          accuracy: undefined,
          credibility: uploadedSource.scores.credibility,
          validity: uploadedSource.scores.validity
        }
      : fallbackScores);

  const iframeSrc = filePath?.startsWith("http") ? filePath : `${API_BASE_URL}${filePath ?? ""}`;

  return (
    <div className="source-detail">
      <header className="source-detail__header">
        <button
          type="button"
          className="source-detail__back-button"
          onClick={() => navigate(-1)}
          aria-label="Back to dashboard"
        >
          ←
        </button>
        <h1 className="source-detail__title">{displayName}</h1>
      </header>
      <div className="source-detail__body">
        <section className="source-detail__viewer card card--viewer">
          <iframe
            src={iframeSrc}
            title={displayName}
            className="source-detail__iframe"
            loading="lazy"
          />
        </section>
        <aside className="source-detail__sidebar">
          <RatingPanel scores={ratingScores} showAccuracy={false} />
          <SummaryPanel
            summary={
              detail?.summary ??
              evaluatedSource?.summary ??
              uploadedSource?.summary ??
              "Summary for this source will appear here once available."
            }
          />
          {detail?.claims?.length ? (
            <article className="card">
              <header className="card__header">
                <h2>Claims Using This Source</h2>
              </header>
              <p className="source-detail__claims-hint">
                These are report claims that cite this document as evidence. Each entry shows the
                latest verdict so you can inspect how the source was used.
              </p>
              <ul className="table-preview">
                {detail.claims.map((claim) => (
                  <li key={claim.claim_id}>
                    <strong>{claim.verdict}</strong>: {claim.text}
                  </li>
                ))}
              </ul>
              {detail.claim_has_more ? (
                <button
                  type="button"
                  className="pill pill--ghost"
                  onClick={() => setClaimPage((prev) => prev + 1)}
                >
                  Show more claims
                </button>
              ) : null}
            </article>
          ) : null}
        </aside>
      </div>
    </div>
  );
};

export default SourceDetail;
