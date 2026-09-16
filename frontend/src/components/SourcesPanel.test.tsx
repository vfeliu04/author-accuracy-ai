import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Report, ReportSource, RunUpload } from "../api/types";
import SourcesPanel from "./SourcesPanel";

const uploads: RunUpload[] = [
  { id: "u1", kind: "REPORT", file_name: "brief.pdf", source_type: "pdf", url: null },
  { id: "u2", kind: "SOURCE", file_name: "ipcc_ch3.pdf", source_type: "pdf", url: null },
  {
    id: "u3",
    kind: "SOURCE",
    file_name: "https://example.org/water/report-2024",
    source_type: "web",
    url: "https://example.org/water/report-2024"
  },
  {
    id: "u4",
    kind: "SOURCE",
    file_name: "https://news.example.com/story?id=9",
    source_type: "web",
    url: "https://news.example.com/story?id=9"
  }
];

function linkUpload(id: string, url: string): RunUpload {
  return { id, kind: "SOURCE", file_name: url, source_type: "web", url };
}

const sources: ReportSource[] = [
  {
    doc_id: "a",
    title: "IPCC Chapter 3",
    source_type: "pdf",
    url: null,
    scorable: true,
    total: 82,
    tier: "VERIFIED_DOI",
    components: { authority: 30 },
    metadata: {}
  },
  {
    doc_id: "b",
    title: "Water report 2024",
    source_type: "web",
    url: "https://example.org/water/report-2024",
    scorable: true,
    total: null,
    tier: null,
    components: null,
    metadata: null
  },
  {
    doc_id: "c",
    title: "chart.png",
    source_type: "image",
    url: null,
    scorable: false,
    total: null,
    tier: null,
    components: null,
    metadata: null
  },
  {
    doc_id: "e",
    title: null,
    source_type: "youtube",
    url: "https://www.youtube.com/watch?v=abc123def45",
    scorable: true,
    total: 41,
    tier: "METADATA_ONLY",
    components: {},
    metadata: {}
  }
];

const doneReport: Report = {
  run_id: "r",
  title: "Brief",
  status: "DONE",
  report_doc_id: "d",
  scores: { accuracy: 1, coverage: 1, credibility: 0.7, validity: 0.5 },
  accuracy_detail: null,
  validity_detail: null,
  credibility_detail: null,
  stats: { claims_total: 3, claims_supported: 1, claims_contradicted: 1, claims_unverifiable: 1 },
  claims: [],
  sources
};

describe("SourcesPanel", () => {
  it("labels each source's type and shows a link by its host and path", () => {
    render(<SourcesPanel uploads={uploads} report={undefined} />);
    expect(screen.getByText("Sources (3)")).toBeInTheDocument();
    expect(screen.getAllByRole("img", { name: "PDF" })).toHaveLength(1);
    expect(screen.getAllByRole("img", { name: "Web page" })).toHaveLength(2);
    expect(screen.getByText("ipcc_ch3.pdf")).toBeInTheDocument();
    expect(screen.getByText("example.org/water/report-2024")).toBeInTheDocument();
    expect(screen.getByText("news.example.com/story")).toHaveAttribute(
      "title",
      "https://news.example.com/story?id=9"
    );
  });

  it("shows links as Queued until the documents are read, then like any other source", () => {
    const { rerender } = render(
      <SourcesPanel uploads={uploads} report={undefined} ingestStatus="running" />
    );
    expect(screen.getAllByText("Queued")).toHaveLength(2);
    expect(screen.getAllByTitle("Received")).toHaveLength(1); // the uploaded file

    rerender(<SourcesPanel uploads={uploads} report={undefined} ingestStatus="done" />);
    expect(screen.queryByText("Queued")).not.toBeInTheDocument();
    expect(screen.getAllByTitle("Received")).toHaveLength(3);
  });

  it("shows links as Queued before any progress exists", () => {
    render(<SourcesPanel uploads={uploads} report={undefined} />);
    expect(screen.getAllByText("Queued")).toHaveLength(2);
  });

  it("flags the link a failed read names, and no other", () => {
    render(
      <SourcesPanel
        uploads={uploads}
        report={undefined}
        ingestStatus="failed"
        runError="FetchError: Fetching 'https://news.example.com/story?id=9' failed: the server answered HTTP 404"
      />
    );
    const flag = screen.getByText("Couldn't open");
    expect(flag).toHaveAttribute("title", expect.stringContaining("The site returned an error (404)."));
    expect(screen.getAllByText("Couldn't open")).toHaveLength(1);
    expect(screen.queryByText("Queued")).not.toBeInTheDocument();
  });

  it("flags a failed link whose address has parentheses, with the reason on hover", () => {
    const wiki = "https://en.wikipedia.org/wiki/Mercury_(planet)";
    render(
      <SourcesPanel
        uploads={[uploads[0], linkUpload("w", wiki)]}
        report={undefined}
        ingestStatus="failed"
        runError={`FetchError: Fetching '${wiki}' failed: the server answered HTTP 404`}
      />
    );
    expect(screen.getByText("Couldn't open")).toHaveAttribute(
      "title",
      `${wiki} — The site returned an error (404).`
    );
  });

  it("gives a link whose site can't be found its reason on hover", () => {
    const link = "https://billing.example.org/report";
    render(
      <SourcesPanel
        uploads={[uploads[0], linkUpload("d", link)]}
        report={undefined}
        ingestStatus="failed"
        runError={`FetchError: Fetching '${link}' failed: 'billing.example.org' could not be resolved ([Errno 8] nodename nor servname provided, or not known)`}
      />
    );
    expect(screen.getByText("Couldn't open")).toHaveAttribute(
      "title",
      `${link} — That site couldn't be found. Check the link for typos, or the internet connection.`
    );
  });

  it("flags only the failed link, not another link its address starts with", () => {
    const report = "https://example.org/report";
    const yearly = "https://example.org/report-2024";
    render(
      <SourcesPanel
        uploads={[uploads[0], linkUpload("a", report), linkUpload("b", yearly)]}
        report={undefined}
        ingestStatus="failed"
        runError={`FetchError: Fetching '${yearly}' failed: the server answered HTTP 404`}
      />
    );
    expect(screen.getAllByText("Couldn't open")).toHaveLength(1);
    const row = screen.getByText("example.org/report-2024").closest(".src-row");
    expect(row).toContainElement(screen.getByText("Couldn't open"));
  });

  it("flags the added link when the failure names where it redirected first", () => {
    render(
      <SourcesPanel
        uploads={uploads}
        report={undefined}
        ingestStatus="failed"
        runError="FetchError: Fetching 'https://www.example.org/water/report-2024/' (redirected from 'https://example.org/water/report-2024') failed: the server answered HTTP 404"
      />
    );
    expect(screen.getAllByText("Couldn't open")).toHaveLength(1);
    const row = screen.getByText("example.org/water/report-2024").closest(".src-row");
    expect(row).toContainElement(screen.getByText("Couldn't open"));
  });

  it("flags a long link the failure message cut short", () => {
    const long = `https://example.org/${"a".repeat(300)}`;
    render(
      <SourcesPanel
        uploads={[uploads[0], linkUpload("l", long)]}
        report={undefined}
        ingestStatus="failed"
        runError={`FetchError: Fetching '${long.slice(0, 200)}...' failed: the server answered HTTP 404`}
      />
    );
    expect(screen.getByText("Couldn't open")).toBeInTheDocument();
  });

  it("keeps listing the uploads until the run is done, even once its documents exist", () => {
    const { rerender } = render(
      <SourcesPanel
        uploads={uploads}
        report={{ ...doneReport, status: "RUNNING", scores: null }}
        ingestStatus="running"
      />
    );
    expect(screen.getByText("Sources (3)")).toBeInTheDocument();
    expect(screen.getAllByText("Queued")).toHaveLength(2);
    expect(screen.queryByText("IPCC Chapter 3")).not.toBeInTheDocument();

    rerender(
      <SourcesPanel
        uploads={uploads}
        report={{ ...doneReport, status: "FAILED", scores: null }}
        ingestStatus="done"
      />
    );
    expect(screen.getByText("Sources (3)")).toBeInTheDocument();
    expect(screen.getAllByTitle("Received")).toHaveLength(3);
    expect(screen.queryByText("IPCC Chapter 3")).not.toBeInTheDocument();
  });

  it("lists every source of a finished report, scored, unscored, and not scorable", () => {
    render(<SourcesPanel uploads={uploads} report={doneReport} ingestStatus="done" />);
    expect(screen.getByText("Sources (4)")).toBeInTheDocument();
    expect(screen.getByText("82")).toBeInTheDocument();
    expect(screen.getByText("verified DOI")).toBeInTheDocument();
    expect(screen.getByText("Water report 2024")).toBeInTheDocument();
    expect(screen.getByText("example.org/water/report-2024")).toBeInTheDocument();
    expect(screen.getByText("Not scored")).toBeInTheDocument();
    expect(screen.getByText("Not scorable")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Image" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Video" })).toBeInTheDocument();
    // An untitled video is named by its link.
    expect(screen.getByText("www.youtube.com/watch")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/NaN|undefined|null/);
  });

  it("opens any source, including one that can't be scored", () => {
    const onOpenSource = vi.fn();
    render(<SourcesPanel uploads={uploads} report={doneReport} onOpenSource={onOpenSource} />);
    fireEvent.click(screen.getByText("chart.png"));
    expect(onOpenSource).toHaveBeenCalledWith("c");
  });
});
