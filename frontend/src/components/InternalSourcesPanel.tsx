import { ChangeEvent, useRef } from "react";
import { InternalSource } from "../data/reportData";

function PdfIcon() {
  return (
    <svg
      width={14}
      height={14}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ flexShrink: 0, opacity: 0.7 }}
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}

type InternalSourcesPanelProps = {
  sources: InternalSource[];
  onSelectSource: (sourceId: string) => void;
  onAddSource: (file: File) => void;
};

// InternalSourcesPanel lists the existing internal sources submitted by the company.
const InternalSourcesPanel = ({ sources, onSelectSource, onAddSource }: InternalSourcesPanelProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onAddSource(file);
    }
    event.target.value = "";
  };

  const handleAddClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <article className="card">
      <header className="card__header">
        <h2>Internal</h2>
      </header>
      <div className="internal-sources">
        <div className="pill-list pill-list--scroll">
          {sources.map((source) => (
            <button
              key={source.id}
              className="pill pill--ghost source-pill"
              type="button"
              onClick={() => onSelectSource(source.id)}
            >
              <PdfIcon />
              <span className="source-pill__name">{source.name}</span>
              {source.usageCount != null && source.usageCount > 0 && (
                <span className="source-pill__badge">{source.usageCount}</span>
              )}
            </button>
          ))}
        </div>
        <div className="pill-list__footer">
          <input
            ref={fileInputRef}
            className="internal-sources__file-input"
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            hidden
          />
          <button className="pill pill--outlined" type="button" onClick={handleAddClick}>
            + Add Internal Source
          </button>
        </div>
      </div>
    </article>
  );
};

export default InternalSourcesPanel;
