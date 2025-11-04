type SummaryPanelProps = {
  summary: string;
};

// SummaryPanel displays the static quality summary for the selected report.
const SummaryPanel = ({ summary }: SummaryPanelProps) => {
  return (
    <article className="card">
      <header className="card__header">
        <h2>Summary</h2>
      </header>
      <p className="card__body-text">{summary}</p>
    </article>
  );
};

export default SummaryPanel;
