import { useSearchParams } from "react-router-dom";
import type { Report, SourceBiblio } from "../api/types";
import { BAND_COLORS, scoreBand } from "../lib/score";
import FocusToolbar from "./FocusToolbar";
import ScoreRing from "./ScoreRing";
import { TIER_LABELS } from "./SourcesPanel";

const COMPONENT_META: Array<{ key: string; label: string; max: number }> = [
  { key: "metadata_completeness", label: "Metadata completeness", max: 30 },
  { key: "authority", label: "Publisher authority", max: 30 },
  { key: "recency", label: "Recency", max: 20 },
  { key: "verification", label: "Verification", max: 20 }
];

const BIBLIO_FIELDS: Array<{ key: keyof SourceBiblio; label: string }> = [
  { key: "title", label: "title" },
  { key: "authors", label: "authors" },
  { key: "publisher", label: "publisher" },
  { key: "publication_date", label: "date" },
  { key: "doi", label: "DOI" }
];

const TIER_EXPLANATIONS: Record<string, string> = {
  VERIFIED_DOI: "The document's own DOI resolved at Crossref — the strongest external confirmation.",
  VERIFIED_TITLE:
    "A Crossref record matched the title exactly, with a second field corroborating it.",
  VERIFIED_ISBN:
    "The ISBN resolved at a book registry and its record corroborates the title or publisher.",
  METADATA_ONLY:
    "Metadata was extracted from the document, but no external registry confirmed it.",
  NONE: "Nothing extractable, so nothing could be verified."
};

function hasField(biblio: SourceBiblio, key: keyof SourceBiblio): boolean {
  const value = biblio[key];
  return Array.isArray(value) ? value.length > 0 : Boolean(value);
}

// Same boundary semantics as the backend's year parser.
function yearOf(date: string | null | undefined): number | null {
  const match = /\b(19|20)\d{2}\b/.exec(date ?? "");
  return match ? Number(match[0]) : null;
}

// The band is fully determined by the stored points — deriving the sentence
// from the value (not a re-computed age) keeps it true forever.
const RECENCY_BANDS: Record<number, string> = {
  20: "under 2 years old",
  12: "2–5 years old",
  6: "5–10 years old",
  3: "over 10 years old"
};

// Credibility is scored by plain code, so every explanation below is DERIVED
// from the same inputs the score used — never prose that could drift.
function explainComponent(
  key: string,
  value: number | undefined,
  biblio: SourceBiblio,
  tier: string
): string | null {
  if (value === undefined) return null;
  if (key === "metadata_completeness") {
    const marks = BIBLIO_FIELDS.map(
      (field) => `${field.label} ${hasField(biblio, field.key) ? "✓" : "✗"}`
    );
    return `6 points per bibliographic field on record: ${marks.join(" · ")}`;
  }
  if (key === "authority") {
    const publisher = biblio.publisher;
    if (value >= 30) return `“${publisher}” is on the tier-1 list of international institutions.`;
    if (value >= 22.5)
      return `“${publisher}” is on the tier-2 list of established outlets and journals.`;
    if (value >= 15)
      return `“${publisher}” is a named publisher, but not on the configured authority lists.`;
    return "No publisher could be found — unknown earns nothing, there are no floors.";
  }
  if (key === "recency") {
    const band = RECENCY_BANDS[value];
    if (!band) return "No usable publication date was found — no points.";
    const year = yearOf(biblio.publication_date);
    return `${year !== null ? `Published ${year} — ` : ""}${band} when scored (20 pts under 2 years, 12 under 5, 6 under 10, 3 older).`;
  }
  if (key === "verification") {
    return TIER_EXPLANATIONS[tier] ?? null;
  }
  return null;
}

// Full-width per-source credibility breakdown. The selected source can be
// deep-linked via ?source=<doc_id> (the sources panel links here).
export default function FocusCredibility({ report }: { report: Report }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const sources = report.sources;
  const selectedParam = searchParams.get("source");
  const selected = sources.find((source) => source.doc_id === selectedParam) ?? sources[0] ?? null;

  const usage =
    report.credibility_detail?.sources?.find((entry) => entry.doc_id === selected?.doc_id)
      ?.usage ?? null;

  const selectSource = (docId: string) => {
    const next = new URLSearchParams(searchParams);
    next.set("source", docId);
    setSearchParams(next, { replace: true });
  };

  return (
    <main className="panel panel--main">
      <FocusToolbar>
        <span className="muted">Credibility · score per source</span>
      </FocusToolbar>
      <div className="claims-split">
        <div className="detail-list">
          {sources.map((source) => (
            <button
              key={source.doc_id}
              type="button"
              className={`claim-item${selected?.doc_id === source.doc_id ? " selected" : ""}`}
              onClick={() => selectSource(source.doc_id)}
            >
              <span
                className={`cred-badge cred-badge--${scoreBand(source.total)}`}
                title="Credibility score"
              >
                {Math.round(source.total)}
              </span>
              <div>
                <div className="claim-item__text">{source.title ?? source.doc_id.slice(0, 8)}</div>
              </div>
            </button>
          ))}
          {sources.length === 0 ? <p className="muted">No sources were scored.</p> : null}
        </div>
        <div className="detail-body">
          {selected ? (
            <div className="detail-inner">
              <div className="focus-summary">
                <ScoreRing value={selected.total / 100} label="Score" size={88} />
                <div className="focus-summary__text">
                  <h3>{selected.title ?? selected.doc_id.slice(0, 8)}</h3>
                  <p>
                    {TIER_LABELS[selected.tier] ?? selected.tier}
                    {usage !== null
                      ? ` · cited by ${usage} verified verdict${usage === 1 ? "" : "s"}`
                      : ""}
                  </p>
                </div>
              </div>
              <div className="rubric-grid">
                {COMPONENT_META.map(({ key, label, max }) => {
                  const value = selected.components[key];
                  const known = value !== undefined;
                  const pct = known ? (value / max) * 100 : 0;
                  const why = explainComponent(key, value, selected.metadata ?? {}, selected.tier);
                  return (
                    <div key={key} className="component-card">
                      <div className="component-card__row">
                        <span className="component-card__name">{label}</span>
                        <span className="component-card__score">
                          {known ? `${Math.round(value * 10) / 10}/${max}` : "—"}
                        </span>
                      </div>
                      <div className="component-bar">
                        <span
                          style={{
                            width: `${Math.min(pct, 100)}%`,
                            background: known ? BAND_COLORS[scoreBand(pct)] : "transparent"
                          }}
                        />
                      </div>
                      {why ? <p className="component-card__text">{why}</p> : null}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <p className="muted">No sources were scored for this run.</p>
          )}
        </div>
      </div>
    </main>
  );
}
