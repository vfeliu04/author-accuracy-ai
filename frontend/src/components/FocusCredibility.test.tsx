import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import type { Report, ReportSource } from "../api/types";
import FocusCredibility from "./FocusCredibility";

const scored: ReportSource = {
  doc_id: "a",
  title: "IPCC Chapter 3",
  source_type: "pdf",
  url: null,
  scorable: true,
  total: 82,
  tier: "VERIFIED_DOI",
  components: { metadata_completeness: 24, authority: 30, recency: 12, verification: 16 },
  metadata: {
    title: "IPCC Chapter 3",
    authors: ["IPCC"],
    publisher: "IPCC",
    publication_date: "2022",
    doi: "10.1017/9781009325844.005"
  },
  truncated: null
};

const unscored: ReportSource = {
  doc_id: "b",
  title: "Water report 2024",
  source_type: "web",
  url: "https://example.org/water",
  scorable: true,
  total: null,
  tier: null,
  components: null,
  metadata: null,
  truncated: null
};

const image: ReportSource = {
  doc_id: "c",
  title: "chart.png",
  source_type: "image",
  url: null,
  scorable: false,
  total: null,
  tier: null,
  components: null,
  metadata: null,
  truncated: null
};

function reportWith(overrides: Partial<Report>): Report {
  return {
    run_id: "r",
    title: "Brief",
    status: "DONE",
    report_doc_id: "d",
    scores: { accuracy: 0.9, coverage: 0.6, credibility: 0.82, validity: 0.5 },
    accuracy_detail: null,
    validity_detail: null,
    credibility_detail: {
      method: "usage_weighted_mean",
      sources: [{ doc_id: "a", total: 82, tier: "VERIFIED_DOI", usage: 3 }],
      excluded: [{ doc_id: "c", reason: "image", usage: 2 }]
    },
    stats: { claims_total: 5, claims_supported: 3, claims_contradicted: 0, claims_unverifiable: 2 },
    claims: [],
    sources: [scored, unscored, image],
    ...overrides
  };
}

function renderAt(report: Report, source?: string) {
  const query = source ? `&source=${source}` : "";
  return render(
    <MemoryRouter initialEntries={[`/runs/r?focus=credibility${query}`]}>
      <FocusCredibility report={report} />
    </MemoryRouter>
  );
}

function aggregate(): HTMLElement {
  return screen.getByText("All sources").closest(".focus-aggregate") as HTMLElement;
}

describe("FocusCredibility", () => {
  it("breaks a scored source down into its components", () => {
    renderAt(reportWith({}), "a");
    expect(screen.getByRole("heading", { name: "IPCC Chapter 3" })).toBeInTheDocument();
    expect(screen.getByText("30/30")).toBeInTheDocument();
    expect(screen.getByText(/verified DOI · cited by 3 verified verdicts/)).toBeInTheDocument();
    expect(aggregate().textContent).toContain("82");
  });

  it("shows a Not scorable card for an image, with how often it is cited", () => {
    renderAt(reportWith({}), "c");
    expect(screen.getByRole("heading", { name: "chart.png" })).toBeInTheDocument();
    expect(screen.getByText("Not scorable")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Images carry no title, author, publisher, or date to check, so they're listed but not counted in credibility."
      )
    ).toBeInTheDocument();
    expect(screen.getByText(/cited by 2 verified verdicts/)).toBeInTheDocument();
    expect(screen.queryByText("Metadata completeness")).not.toBeInTheDocument();
  });

  it("shows a Not scored card for a scorable source without a score", () => {
    renderAt(reportWith({}), "b");
    expect(screen.getByRole("heading", { name: "Water report 2024" })).toBeInTheDocument();
    expect(screen.getByText("Not scored")).toBeInTheDocument();
    expect(screen.queryByText("Metadata completeness")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/NaN|undefined/);
  });

  it("lists unscored sources without a number", () => {
    renderAt(reportWith({}), "a");
    const badges = Array.from(document.querySelectorAll(".detail-list .cred-badge"));
    expect(badges.map((badge) => badge.textContent)).toEqual(["82", "—", "—"]);
    expect(badges[1]).toHaveAttribute("title", "Not scored");
    expect(badges[2]).toHaveAttribute("title", "Not scorable");
  });

  it("says what a matched registry record did and did not confirm", () => {
    // A publisher's own article page: the record corroborates it, but the page
    // is not the address that record names, so it is neither verified nor a
    // source nothing matched.
    const matched: ReportSource = {
      ...scored,
      doc_id: "a",
      tier: "MATCHED_RECORD",
      components: { metadata_completeness: 30, authority: 30, recency: 20, verification: 10 }
    };
    renderAt(reportWith({ sources: [matched] }), "a");
    expect(screen.getByText("10/20")).toBeInTheDocument();
    expect(
      screen.getByText(
        "A registry record matches this document's details, but it doesn't name the address this was fetched from — so it could not be confirmed to be that work."
      )
    ).toBeInTheDocument();
    expect(screen.getByText(/registry match only/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("MATCHED_RECORD");
  });

  it("keeps the unverified sentence for a source no registry record matched", () => {
    const floor: ReportSource = {
      ...scored,
      tier: "METADATA_ONLY",
      components: { metadata_completeness: 30, authority: 30, recency: 20, verification: 5 }
    };
    renderAt(reportWith({ sources: [floor] }), "a");
    expect(
      screen.getByText(
        "Metadata was extracted from the document, and no registry record matched it."
      )
    ).toBeInTheDocument();
  });

  it("renders a missing run credibility as a dash with a short reason, never a number", () => {
    renderAt(
      reportWith({
        scores: { accuracy: 0.9, coverage: 0.6, credibility: null, validity: 0.5 },
        credibility_detail: {
          method: "no_scorable_sources",
          sources: [],
          excluded: [{ doc_id: "c", reason: "image", usage: 2 }]
        },
        sources: [image]
      })
    );
    const text = aggregate().textContent ?? "";
    expect(text).toContain("—");
    expect(text).toContain("No scorable sources");
    expect(text).not.toMatch(/\d/);
    expect(screen.getByText("Not scorable")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/NaN|undefined/);
  });

  it("says a run with no sources has none to score, instead of a number", () => {
    renderAt(
      reportWith({
        scores: { accuracy: 0.9, coverage: 0.6, credibility: null, validity: 0.5 },
        credibility_detail: { method: "no_sources", sources: [], excluded: [] },
        sources: []
      })
    );
    const text = aggregate().textContent ?? "";
    expect(text).toContain("—");
    expect(text).toContain("No sources to score");
    expect(text).not.toMatch(/\d/);
    expect(document.body.textContent).not.toMatch(/NaN|undefined/);
  });

  it("stays null-safe for a run without stored credibility detail", () => {
    renderAt(reportWith({ scores: null, credibility_detail: null, sources: [unscored] }));
    expect(aggregate().textContent).toContain("—");
    expect(screen.getAllByText("Not scored").length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toMatch(/NaN|undefined/);
  });
});
