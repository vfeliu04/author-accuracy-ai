import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useCreateRun } from "../api/queries";
import { formatBytes } from "../lib/format";

// Client-side mirrors of the server caps — fail fast in the dialog instead
// of after a full upload (the server remains the authority).
const MAX_SOURCES = 20;
const MAX_FILE_BYTES = 50_000_000;
const MAX_TOTAL_BYTES = 200_000_000;

function stem(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
}

function isPdf(file: File): boolean {
  return file.name.toLowerCase().endsWith(".pdf");
}

export default function UploadDialog({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const create = useCreateRun();
  const [report, setReport] = useState<File | null>(null);
  const [sources, setSources] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const addFiles = (list: FileList | File[]) => {
    setError(null);
    const incoming = Array.from(list);
    const nonPdf = incoming.find((file) => !isPdf(file));
    const oversize = incoming.find((file) => isPdf(file) && file.size > MAX_FILE_BYTES);
    if (nonPdf) {
      setError(`Only PDF files can be verified — “${nonPdf.name}” was not added.`);
    } else if (oversize) {
      setError(`“${oversize.name}” is over the ${formatBytes(MAX_FILE_BYTES)} per-file limit.`);
    }
    const accepted = incoming.filter((file) => isPdf(file) && file.size <= MAX_FILE_BYTES);
    if (accepted.length === 0) return;
    if (report === null) {
      const [first, ...rest] = accepted;
      setReport(first);
      setSources([...sources, ...rest]);
      if (!nameTouched) setName(stem(first.name));
    } else {
      setSources([...sources, ...accepted]);
    }
  };

  const removeReport = () => {
    setReport(null);
    if (!nameTouched) setName("");
  };

  const removeSource = (index: number) => {
    setSources(sources.filter((_, i) => i !== index));
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    addFiles(event.dataTransfer.files);
  };

  const totalBytes = (report?.size ?? 0) + sources.reduce((sum, file) => sum + file.size, 0);
  const fileCount = (report ? 1 : 0) + sources.length;
  const tooManySources = sources.length > MAX_SOURCES;
  const tooBig = totalBytes > MAX_TOTAL_BYTES;
  const canSubmit =
    report !== null && sources.length > 0 && !tooManySources && !tooBig && !create.isPending;

  const submit = () => {
    if (!report) return;
    create.mutate(
      { report, sources, title: name },
      {
        onSuccess: (data) => {
          onClose();
          navigate(`/runs/${data.run_id}`);
        },
        onError: (err) => setError(err instanceof Error ? err.message : "The upload failed.")
      }
    );
  };

  return (
    <div
      className="modal-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal" role="dialog" aria-modal="true" aria-label="New verification">
        <div className="modal__head">
          <h2>New verification</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="modal__body">
          <label className="field-label" htmlFor="run-name">
            Name
          </label>
          <input
            id="run-name"
            className="name-input"
            value={name}
            maxLength={200}
            placeholder="Named after the report unless you change it"
            onChange={(event) => {
              setName(event.target.value);
              setNameTouched(true);
            }}
          />

          <span className="field-label">Report under review</span>
          {report ? (
            <div className="file-row file-row--report">
              <span className="file-row__icon" aria-hidden>
                📄
              </span>
              <span className="file-row__name">{report.name}</span>
              <span className="file-row__size">{formatBytes(report.size)}</span>
              <button
                type="button"
                className="file-row__remove"
                onClick={removeReport}
                aria-label={`Remove ${report.name}`}
              >
                ✕
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="file-slot"
              onClick={() => inputRef.current?.click()}
            >
              <span className="file-row__icon" aria-hidden>
                📄
              </span>
              The report PDF — the first file you add lands here
            </button>
          )}

          <span className="field-label">Sources ({sources.length})</span>
          {sources.length === 0 ? (
            <button
              type="button"
              className="file-slot"
              onClick={() => inputRef.current?.click()}
            >
              <span className="file-row__icon" aria-hidden>
                📘
              </span>
              The source PDFs the report will be checked against
            </button>
          ) : null}
          {sources.map((file, index) => (
            <div className="file-row" key={`${file.name}-${index}`}>
              <span className="file-row__icon" aria-hidden>
                📘
              </span>
              <span className="file-row__name">{file.name}</span>
              <span className="file-row__size">{formatBytes(file.size)}</span>
              <button
                type="button"
                className="file-row__remove"
                onClick={() => removeSource(index)}
                aria-label={`Remove ${file.name}`}
              >
                ✕
              </button>
            </div>
          ))}

          <button
            type="button"
            className={`dropzone${dragOver ? " dropzone--active" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            {report === null ? (
              <>
                Drop the report PDF here, then its sources — or <b>browse</b>
              </>
            ) : (
              <>
                Drag source PDFs here or <b>browse</b>
              </>
            )}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            multiple
            hidden
            onChange={(event) => {
              if (event.target.files) addFiles(event.target.files);
              event.target.value = "";
            }}
          />

          {error ? <p className="modal__error">{error}</p> : null}
          {tooManySources ? (
            <p className="modal__error">At most {MAX_SOURCES} sources per verification.</p>
          ) : null}
          {tooBig ? (
            <p className="modal__error">
              Total upload is over {formatBytes(MAX_TOTAL_BYTES)} — remove some files.
            </p>
          ) : null}
        </div>
        <div className="modal__foot">
          <span className="modal__count">
            {fileCount === 0
              ? "No files yet"
              : `${fileCount} file${fileCount === 1 ? "" : "s"} · ${formatBytes(totalBytes)}`}
          </span>
          <div className="modal__actions">
            <button type="button" className="btn btn--ghost" onClick={onClose}>
              Cancel
            </button>
            <button type="button" className="btn btn--primary" disabled={!canSubmit} onClick={submit}>
              {create.isPending ? "Uploading…" : "Verify report"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
