import { ChangeEvent, useRef } from "react";
import { InternalSource } from "../data/reportData";

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
              className="pill pill--ghost"
              type="button"
              onClick={() => onSelectSource(source.id)}
            >
              {source.name}
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
