type RecommendedSourcesPanelProps = {
  sources: string[];
};

// RecommendedSourcesPanel lists alternative external sources suggested by the system.
const RecommendedSourcesPanel = ({ sources }: RecommendedSourcesPanelProps) => {
  return (
    <article className="card">
      <header className="card__header">
        <h2>Recommended</h2>
      </header>
      <div className="pill-list pill-list--scroll">
        {sources.map((source) => (
          <button key={source} className="pill pill--filled" type="button">
            {source}
          </button>
        ))}
      </div>
    </article>
  );
};

export default RecommendedSourcesPanel;
