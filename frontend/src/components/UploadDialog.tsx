import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useCreateRun, useReferenceScan } from "../api/queries";
import type { LookupStatus, ReferenceScan, ScannedReference } from "../api/types";
import { formatBytes, plural } from "../lib/format";
import { checkLink, linkHost, linkHostPath } from "../lib/links";
import { alreadyAdded, stem } from "../lib/references";

// Client-side mirrors of the server caps — fail fast in the dialog instead
// of after a full upload (the server remains the authority). Files and links
// share the source cap.
const MAX_SOURCES = 20;
const MAX_FILE_BYTES = 50_000_000;
const MAX_TOTAL_BYTES = 200_000_000;

function isPdf(file: File): boolean {
  return file.name.toLowerCase().endsWith(".pdf");
}

// The server remains the authority: a link the dialog's checks let through can
// still be refused, in wording written for the log, so it becomes one sentence.
function uploadError(err: unknown): string {
  if (!(err instanceof Error)) return "The upload failed.";
  return err.message.startsWith("not a usable link:")
    ? "One of the links isn't a valid web address. Check it and try again."
    : err.message;
}

// A cited work with its place in the scan, which is what a tick refers to.
type Cited = { ref: ScannedReference; index: number };

// A free copy the lookup found is worth adding; an address the entry merely
// printed was validated but never visited, so it waits for a tick.
function foundCopy(ref: ScannedReference): boolean {
  return ref.suggested_url !== null && (ref.retrievability === "pdf" || ref.retrievability === "landing");
}

function defaultTicks(scan: ReferenceScan | undefined): ReadonlySet<number> {
  const ticks = new Set<number>();
  scan?.references.forEach((ref, index) => {
    if (foundCopy(ref)) ticks.add(index);
  });
  return ticks;
}

// What the row says about a copy of the work.
function copyTag(ref: ScannedReference, lookup: LookupStatus): string {
  switch (ref.retrievability) {
    case "pdf":
      return "free PDF";
    case "landing":
      return "free copy";
    case "paywalled":
      return "paywalled";
    default:
      if (ref.suggested_url !== null) return "printed in the entry, not checked";
      if (lookup === "unconfigured") return "lookup not set up";
      if (lookup === "unavailable") return "lookup unavailable";
      return "no link found";
  }
}

// A scan the dialog itself dropped (report swapped, dialog closed) is not
// something to tell the reader about.
function scanMessage(error: Error | null): string | null {
  return error !== null && error.name !== "AbortError" ? error.message : null;
}

// What the scan's caps cut, in one line: a list read or kept in part must not
// pass for the whole one, since the works past the cut are unread, not absent.
function limitsNote({ text_truncated, references_dropped }: ReferenceScan["limits"]): string | null {
  const parts = [
    text_truncated ? "Read the first part of a long reference list" : null,
    references_dropped > 0
      ? `${references_dropped === 1 ? "1 entry" : `${references_dropped} entries`} not shown`
      : null
  ].filter((part): part is string => part !== null);
  return parts.length === 0 ? null : parts.join(" — ");
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

  // The report's own reference list, read as soon as the report is picked.
  // Which rows are ticked belongs to one scan: a set made for another answer
  // is ignored and the defaults for this one stand, with no frame in between.
  const scan = useReferenceScan(report);
  const [ticks, setTicks] = useState<{ of: ReferenceScan; set: ReadonlySet<number> } | null>(null);
  // What the last Add click could not do, about this scan's rows: gone with
  // the report it was about, like the ticks.
  const [addNote, setAddNote] = useState<{ of: ReferenceScan; text: string } | null>(null);
  const defaults = useMemo(() => defaultTicks(scan.data), [scan.data]);
  const checked = ticks !== null && ticks.of === scan.data ? ticks.set : defaults;
  const note = addNote !== null && addNote.of === scan.data ? addNote.text : null;

  // Matched on every render: sources arrive after the report, so the answer
  // is always taken against what is in the dialog now.
  const missing = useMemo<Cited[]>(() => {
    const names = sources.map((file) => file.name);
    return (scan.data?.references ?? [])
      .map((ref, index) => ({ ref, index }))
      .filter(({ ref }) => alreadyAdded(ref, names, links) === null);
  }, [scan.data, sources, links]);
  const addable = useMemo(() => missing.filter(({ ref }) => ref.suggested_url !== null), [missing]);
  const ticked = addable.filter(({ index }) => checked.has(index));

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

  const toggle = (index: number) => {
    if (!scan.data) return;
    const next = new Set(checked);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    setTicks({ of: scan.data, set: next });
  };

  // The only way a suggestion becomes a source: the ticked rows, each through
  // the same checks a typed link gets, up to the source cap. A duplicate is
  // skipped (its row goes once the first copy is in); the cap stops the rest
  // and says so, with what got in.
  const addSuggested = () => {
    if (!scan.data) return;
    const wanted = ticked.map(({ ref }) => ref.suggested_url as string);
    let next = links;
    let added = 0;
    let capped = false;
    for (const url of wanted) {
      if (sources.length + next.length >= MAX_SOURCES) {
        capped = true;
        break;
      }
      const result = checkLink(url, next);
      if ("error" in result) continue;
      next = [...next, result.link];
      added += 1;
    }
    setLinks(next);
    setAddNote(
      capped
        ? {
            of: scan.data,
            text: `Added ${added} of ${wanted.length} — at most ${MAX_SOURCES} sources per verification.`
          }
        : null
    );
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    addFiles(event.dataTransfer.files);
  };

  const totalBytes = (report?.size ?? 0) + sources.reduce((sum, file) => sum + file.size, 0);
  const fileCount = (report ? 1 : 0) + sources.length;
  const sourceCount = sources.length + links.length;
  // A link still in the box counts: submit() adds it first, or holds and says why.
  const hasTypedLink = linkText.trim() !== "";
  const tooManySources = sourceCount > MAX_SOURCES;
  const tooBig = totalBytes > MAX_TOTAL_BYTES;
  const canSubmit =
    report !== null &&
    (sourceCount > 0 || hasTypedLink) &&
    !tooManySources &&
    !tooBig &&
    !create.isPending;

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
        onError: (err) => setError(uploadError(err))
      }
    );
  };

  const scanError = scanMessage(scan.error);
  const lookup = scan.data?.lookup;
  const cut = scan.data ? limitsNote(scan.data.limits) : null;

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
                {linkHostPath(link)}
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

          {report !== null && scanError === null ? (
            <span className="field-label">
              Cited by the report, not among your sources
              {scan.data ? ` (${missing.length})` : ""}
            </span>
          ) : null}
          {scan.isLoading ? (
            <p className="modal__count">Scanning the report's references…</p>
          ) : null}
          {scanError !== null ? <p className="modal__error">{scanError}</p> : null}
          {scan.data && scan.data.references.length === 0 ? (
            <p className="modal__count">No reference list found in this report</p>
          ) : null}
          {scan.data && scan.data.references.length > 0 && missing.length === 0 ? (
            <p className="modal__count">Every cited work is among your sources.</p>
          ) : null}
          {lookup?.status === "unavailable" ? (
            <p className="modal__count">
              Free copies could not be looked up{lookup.detail ? `: ${lookup.detail}` : "."}
            </p>
          ) : null}
          {cut !== null ? <p className="modal__count">{cut}</p> : null}
          {missing.map(({ ref, index }) => {
            // The title, else the printed text; a row the scan kept for its
            // DOI or address alone, with no text, is named by that. Never a
            // blank label or tooltip.
            const label = ref.title ?? (ref.entry || ref.doi || ref.url || "(untitled entry)");
            const tooltip = ref.entry || label;
            const tag = copyTag(ref, lookup?.status ?? "ok");
            if (ref.suggested_url === null) {
              return (
                <div className="file-row" key={index} title={tooltip}>
                  <span className="file-row__check" aria-hidden />
                  <span className="file-row__name">{label}</span>
                  <span className="file-row__size">{tag}</span>
                </div>
              );
            }
            return (
              <label className="file-row" key={index} title={tooltip}>
                <input
                  type="checkbox"
                  className="file-row__check"
                  checked={checked.has(index)}
                  onChange={() => toggle(index)}
                />
                <span className="file-row__name">{label}</span>
                <span className="file-row__host" title={ref.suggested_url}>
                  {linkHost(ref.suggested_url)}
                </span>
                <span className="file-row__size">{tag}</span>
              </label>
            );
          })}
          {addable.length > 0 ? (
            <button
              type="button"
              className="btn btn--ghost"
              disabled={ticked.length === 0}
              onClick={addSuggested}
            >
              Add {plural(ticked.length, "link")}
            </button>
          ) : null}
          {note !== null ? <p className="modal__count">{note}</p> : null}

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
