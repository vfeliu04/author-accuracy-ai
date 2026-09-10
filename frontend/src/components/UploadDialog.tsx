import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useCreateRun } from "../api/queries";
import { formatBytes } from "../lib/format";
import { checkLink, linkHost } from "../lib/links";

// Client-side mirrors of the server caps — fail fast in the dialog instead
// of after a full upload (the server remains the authority). Files and links
// share the source cap.
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

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export default function UploadDialog({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const create = useCreateRun();
  const [report, setReport] = useState<File | null>(null);
  const [sources, setSources] = useState<File[]>([]);
  const [links, setLinks] = useState<string[]>([]);
  const [linkText, setLinkText] = useState("");
  const [linkError, setLinkError] = useState<string | null>(null);
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

  // Adds the link in the box and returns the new list. A rejected link stays in
  // the box, with the reason, so it can be corrected in place.
  const commitLink = (): string[] | null => {
    const result = checkLink(linkText, links);
    if ("error" in result) {
      setLinkError(result.error);
      return null;
    }
    const next = [...links, result.link];
    setLinks(next);
    setLinkText("");
    setLinkError(null);
    return next;
  };

  const addLink = () => {
    if (linkText.trim() !== "") commitLink();
  };

  const removeReport = () => {
    setReport(null);
    if (!nameTouched) setName("");
  };

  const removeSource = (index: number) => {
    setSources(sources.filter((_, i) => i !== index));
  };

  const removeLink = (link: string) => {
    setLinks(links.filter((added) => added !== link));
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    addFiles(event.dataTransfer.files);
  };

  const totalBytes = (report?.size ?? 0) + sources.reduce((sum, file) => sum + file.size, 0);
  const fileCount = (report ? 1 : 0) + sources.length;
  const sourceCount = sources.length + links.length;
  const tooManySources = sourceCount > MAX_SOURCES;
  const tooBig = totalBytes > MAX_TOTAL_BYTES;
  const canSubmit =
    report !== null && sourceCount > 0 && !tooManySources && !tooBig && !create.isPending;

  const countParts = [
    fileCount > 0 ? plural(fileCount, "file") : null,
    links.length > 0 ? plural(links.length, "link") : null,
    fileCount > 0 ? formatBytes(totalBytes) : null
  ].filter((part): part is string => part !== null);

  const submit = () => {
    if (!report) return;
    // A link still in the box was meant to go too, so it is added first; one
    // that can't be added, or that passes the limit, holds the upload.
    let submitted = links;
    if (linkText.trim() !== "") {
      const next = commitLink();
      if (next === null || sources.length + next.length > MAX_SOURCES) return;
      submitted = next;
    }
    create.mutate(
      { report, sources, links: submitted, title: name },
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

          <span className="field-label">Sources ({sourceCount})</span>
          {sourceCount === 0 ? (
            <button
              type="button"
              className="file-slot"
              onClick={() => inputRef.current?.click()}
            >
              <span className="file-row__icon" aria-hidden>
                📘
              </span>
              The source PDFs the report will be checked against — or add links below
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
          {links.map((link) => (
            <div className="file-row" key={link}>
              <span className="file-row__icon" aria-hidden>
                🔗
              </span>
              <span className="file-row__name file-row__host" title={link}>
                {linkHost(link)}
              </span>
              <button
                type="button"
                className="file-row__remove"
                onClick={() => removeLink(link)}
                aria-label={`Remove ${link}`}
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

          <label className="field-label" htmlFor="source-link">
            Add a link
          </label>
          <div className="link-add">
            <input
              id="source-link"
              className="name-input link-add__input"
              type="url"
              inputMode="url"
              autoComplete="off"
              spellCheck={false}
              placeholder="https://"
              value={linkText}
              aria-invalid={linkError ? true : undefined}
              aria-describedby={linkError ? "source-link-error" : undefined}
              onChange={(event) => {
                setLinkText(event.target.value);
                setLinkError(null);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  addLink();
                }
              }}
            />
            <button
              type="button"
              className="btn btn--ghost"
              disabled={linkText.trim() === ""}
              onClick={addLink}
            >
              Add
            </button>
          </div>
          {linkError ? (
            <p id="source-link-error" className="modal__error">
              {linkError}
            </p>
          ) : null}

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
            {countParts.length === 0 ? "Nothing added yet" : countParts.join(" · ")}
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
