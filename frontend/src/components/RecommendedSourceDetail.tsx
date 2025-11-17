import { useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useReportData } from "../context/ReportDataContext";

const RecommendedSourceDetail = () => {
  const navigate = useNavigate();
  const { sourceIndex } = useParams<{ sourceIndex: string }>();
  const { summaryData } = useReportData();
  const recommendations = summaryData?.recommended_sources ?? [];

  const entry = useMemo(() => {
    if (!sourceIndex) {
      return null;
    }
    const idx = Number(sourceIndex);
    if (Number.isNaN(idx)) {
      return null;
    }
    return recommendations[idx] ?? null;
  }, [recommendations, sourceIndex]);

  if (!entry) {
    return (
      <div className="source-detail source-detail--missing">
        <div className="source-detail__missing-card">
          <p>Recommendation details are unavailable. Run the pipeline and open a recommendation again.</p>
          <button type="button" className="pill pill--filled" onClick={() => navigate("/dashboard")}>
            Return to dashboard
          </button>
        </div>
      </div>
    );
  }

  const authors = entry.authors?.length ? entry.authors.join(", ") : "Unknown authors";
  const venueParts = [
    entry.host_venue,
    entry.publication_year ? `Published ${entry.publication_year}` : null
  ].filter(Boolean);
  const metadata: Array<{ label: string; value: string }> = [
    {
      label: "Date published",
      value: entry.date_published ?? (venueParts.join(" · ") || "Information missing")
    },
    { label: "Authors", value: authors || "Information missing" },
    { label: "DOI", value: entry.doi ?? "Information missing" },
    { label: "OpenAlex link", value: entry.openalex_url ? "Available" : "Information missing" }
  ];
  const abstractText = entry.abstract?.trim();
  const summaryText = entry.summary ?? abstractText ?? "Summary information missing.";
  const credibilityPercent =
    typeof entry.credibility_score === "number" ? Math.round(entry.credibility_score) : null;
  const validityPercent = typeof entry.validity_score === "number" ? Math.round(entry.validity_score) : null;
  const credibilityDisplay = credibilityPercent !== null ? `${credibilityPercent}%` : "Information missing";
  const validityDisplay = validityPercent !== null ? `${validityPercent}%` : "Information missing";
  const ratingRows = [
    { label: "Credibility", display: credibilityDisplay, percent: credibilityPercent ?? 0 },
    { label: "Validity", display: validityDisplay, percent: validityPercent ?? 0 }
  ];

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
        <h1 className="source-detail__title">{entry.title}</h1>
      </header>
      <div className="source-detail__body source-detail__body--single">
        <section className="recommended-layout">
          <div className="recommended-panel recommended-panel--primary">
            <div className="recommended-block">
              <h3>Summary</h3>
              <p>{summaryText}</p>
            </div>
            <div className="recommended-block">
              <h3>Abstract</h3>
              <p>{abstractText || "Abstract information missing."}</p>
            </div>
          </div>
          <div className="recommended-panel recommended-panel--side">
            <div className="recommended-group">
              <h4>Ratings</h4>
              <div className="recommended-rating">
                {ratingRows.map((row) => (
                  <div className="recommended-rating__row" key={row.label}>
                    <span
                      className="recommended-rating__icon"
                      style={{
                        background: `conic-gradient(#6366f1 0% ${row.percent}%, #e2e8f0 ${row.percent}% 100%)`
                      }}
                    >
                      <span className="recommended-rating__icon-inner" />
                    </span>
                    <div className="recommended-rating__info">
                      <span className="recommended-rating__label">{row.label}</span>
                      <span className="recommended-rating__value">{row.display}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="recommended-group">
              <h4>Information</h4>
              <ul>
                {metadata.map((item) => (
                  <li key={item.label}>
                    <strong>{item.label}</strong>
                    <span>{item.value}</span>
                  </li>
                ))}
              </ul>
              <div className="recommended-links">
                {entry.url ? (
                  <a className="pill pill--filled recommended-detail__link" href={entry.url} target="_blank" rel="noreferrer">
                    Open source
                  </a>
                ) : null}
                {entry.openalex_url ? (
                  <a
                    className="pill pill--ghost recommended-detail__link"
                    href={entry.openalex_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    View on OpenAlex
                  </a>
                ) : null}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default RecommendedSourceDetail;
