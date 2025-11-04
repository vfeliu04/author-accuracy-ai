type InternalSourcesPanelProps = {
  sources: string[];
};

// InternalSourcesPanel lists the existing internal sources submitted by the company.
const InternalSourcesPanel = ({ sources }: InternalSourcesPanelProps) => {
  return (
    <article className="card">
      <header className="card__header">
        <h2>Internal</h2>
      </header>
      <div className="pill-list pill-list--scroll">
        {sources.map((source) => (
          <button key={source} className="pill pill--ghost" type="button">
            {source}
          </button>
        ))}
      </div>
    </article>
  );
};

export default InternalSourcesPanel;
