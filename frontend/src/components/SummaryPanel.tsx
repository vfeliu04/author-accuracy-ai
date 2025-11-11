type SummaryStats = {
  claims_total: number;
  claims_supported: number;
  claims_contradicted: number;
  claims_not_found: number;
};

type SummaryPanelProps = {
  summary?: string;
  reportLabel?: string;
  onOpenReport?: () => void;
  stats?: SummaryStats;
  topSources?: Array<{ name: string; usage_count: number }>;
};

// SummaryPanel displays either the analytics snapshot or a text summary if stats are unavailable.
const SummaryPanel = ({ summary, reportLabel, onOpenReport, stats, topSources }: SummaryPanelProps) => {
  const hasStats = Boolean(stats);

  return (
    <article className="card card--summary">
      <header className="card__header">
        <h2>Summary</h2>
      </header>
      {reportLabel && onOpenReport ? (
        <div className="summary__report-launch">
          <button type="button" className="summary__report-button" onClick={onOpenReport}>
            {reportLabel}
          </button>
        </div>
      ) : null}

      {hasStats ? (
        <div className="summary__scrollable">
          {summary ? (
            <div className="summary__text-block">
              <p>{summary}</p>
            </div>
          ) : null}
          <div className="summary__analytics">
            <div className="summary__analytics-row">
              <div className="summary__stat">
                <span>Total Claims</span>
                <strong>{stats?.claims_total ?? 0}</strong>
              </div>
              <div className="summary__stat">
                <span>Supported</span>
                <strong>{stats?.claims_supported ?? 0}</strong>
              </div>
            </div>
            <div className="summary__analytics-row">
              <div className="summary__stat">
                <span>Contradicted</span>
                <strong>{stats?.claims_contradicted ?? 0}</strong>
              </div>
              <div className="summary__stat">
                <span>Not Found</span>
                <strong>{stats?.claims_not_found ?? 0}</strong>
              </div>
            </div>
          </div>
          {topSources?.length ? (
            <div className="summary__claims">
              <h3>Claims Supported</h3>
              <ul>
                {topSources.map((source) => (
                  <li key={source.name}>
                    <span>{source.name}</span>
                    <strong>{source.usage_count} claims</strong>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="summary__content">
          <p className="card__body-text summary__text">{summary ?? "Summary unavailable."}</p>
        </div>
      )}
    </article>
  );
};

export default SummaryPanel;
