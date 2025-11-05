type SummaryPanelProps = {
  summary: string;
  reportLabel?: string;
  onOpenReport?: () => void;
};

// SummaryPanel displays the static quality summary for the selected report.
const SummaryPanel = ({ summary, reportLabel, onOpenReport }: SummaryPanelProps) => {
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
      <div className="summary__content">
        <p className="card__body-text summary__text">{summary}</p>
      </div>
    </article>
  );
};

export default SummaryPanel;
