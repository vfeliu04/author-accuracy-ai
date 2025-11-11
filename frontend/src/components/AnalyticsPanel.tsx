type AnalyticsPanelProps = {
  stats: {
    claims_total: number;
    claims_supported: number;
    claims_contradicted: number;
    claims_not_found: number;
  };
  topSources: Array<{ id: string; name: string; usage_count: number; credibility: number }>;
};

const HARDCODED_RECOMMENDATIONS = [
  "Global Food Resilience Index 2025",
  "Nutrition Equity Observatory Brief",
  "FAO Logistics Pulse",
  "World Bank Commodity Outlook",
  "WFP Supply Chain Preparedness Guide"
];

const AnalyticsPanel = (_props: AnalyticsPanelProps) => {
  return (
    <article className="card">
      <header className="card__header">
        <h2>Recommended Sources</h2>
      </header>
      <div className="analytics__sources analytics__sources--stacked">
        <ul>
          {HARDCODED_RECOMMENDATIONS.map((source) => (
            <li key={source}>
              <div className="analytics__source">
                <span>{source}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
};

export default AnalyticsPanel;
