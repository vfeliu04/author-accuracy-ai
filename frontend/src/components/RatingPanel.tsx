type RatingPanelProps = {
  scores: {
    overall: number;
    accuracy: number;
    credibility: number;
    validity: number;
  };
};

// RatingPanel showcases the overall score bar and individual metric snapshots.
const RatingPanel = ({ scores }: RatingPanelProps) => {
  const metricDetails = [
    { label: "Accuracy", value: scores.accuracy },
    { label: "Credibility", value: scores.credibility },
    { label: "Validity", value: scores.validity }
  ];

  return (
    <article className="card">
      <header className="card__header">
        <h2>Rating</h2>
      </header>
      <div className="rating__overall">
        <span className="rating__label">Overall</span>
        <div className="rating__bar">
          <div
            className="rating__bar-fill"
            style={{ width: `${Math.round(scores.overall * 100)}%` }}
          />
        </div>
        <span className="rating__score">{(scores.overall * 100).toFixed(0)}%</span>
      </div>
      <div className="rating__metrics">
        {metricDetails.map((metric) => {
          const percentage = Math.round(metric.value * 100);
          const iconStyle = {
            background: `conic-gradient(#6366f1 0% ${percentage}%, #e2e8f0 ${percentage}% 100%)`
          };

          return (
            <div className="rating__metric" key={metric.label}>
              <span className="rating__metric-icon" style={iconStyle}>
                <span className="rating__metric-icon-inner" />
              </span>
              <div className="rating__metric-info">
                <span className="rating__metric-label">{metric.label}</span>
                <span className="rating__metric-score">{percentage}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
};

export default RatingPanel;
