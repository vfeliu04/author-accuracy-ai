import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EvidenceSource, PageSnapshot } from "../api/types";
import * as v2 from "../api/v2";
import SourcePane from "./SourcePane";

const pdfSource: EvidenceSource = {
  doc_id: "d",
  title: "Source",
  page: null,
  source_type: "pdf",
  url: null,
  section: null,
  start_seconds: null,
  chunk_id: 1
};

const page: PageSnapshot = {
  schema: 1,
  document: {
    title: "Water in Crisis",
    sections: [{ title: "Findings", page: null, text: "People lack safe water." }]
  },
  provenance: {
    url: "https://example.org/water",
    final_url: "https://example.org/water",
    fetched_at: "2026-09-01T10:00:00Z",
    content_type: "text/html",
    title: "Water in Crisis",
    authors: [],
    publisher: null,
    publication_date: null,
    doi: null,
    scholarly: false
  }
};

function renderPane(source: EvidenceSource, quote: string | null = null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <SourcePane runId="r" source={source} quote={quote} />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  // jsdom has no object-URL support; the PDF viewer shows blobs through one.
  (URL as unknown as { createObjectURL: unknown }).createObjectURL = vi.fn(() => "blob:fake");
  (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = vi.fn();
});

afterEach(() => vi.restoreAllMocks());

describe("SourcePane", () => {
  it("shows a PDF source in the PDF viewer at the cited page", async () => {
    const blob = vi
      .spyOn(v2, "fetchPdfBlob")
      .mockResolvedValue(new Blob(["%PDF"], { type: "application/pdf" }));
    const json = vi.spyOn(v2, "fetchDocumentJson");
    renderPane({ ...pdfSource, page: 4 });
    const frame = await screen.findByTitle("source");
    expect(frame).toHaveAttribute("src", "blob:fake#page=4&toolbar=0&navpanes=0&view=FitH");
    expect(blob).toHaveBeenCalledWith("r", "d");
    expect(json).not.toHaveBeenCalled();
  });

  it("shows a web source as readable text with the quote marked", async () => {
    const json = vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const blob = vi.spyOn(v2, "fetchPdfBlob");
    const { container } = renderPane(
      { ...pdfSource, source_type: "web", url: "https://example.org/water", section: "Findings" },
      "lack safe water"
    );
    expect(await screen.findByRole("heading", { name: "Water in Crisis" })).toBeInTheDocument();
    expect(container.querySelector("mark")?.textContent).toBe("lack safe water");
    expect(json).toHaveBeenCalledWith("r", "d");
    expect(blob).not.toHaveBeenCalled();
    expect(screen.queryByTitle("source")).not.toBeInTheDocument();
  });

  it("offers the original of an image instead of a preview", () => {
    const blob = vi.spyOn(v2, "fetchPdfBlob");
    const json = vi.spyOn(v2, "fetchDocumentJson");
    renderPane({ ...pdfSource, source_type: "image", url: "https://example.org/chart.png" });
    expect(screen.getByText(/preview isn't available for this image yet/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open original ↗" })).toHaveAttribute(
      "href",
      "https://example.org/chart.png"
    );
    expect(blob).not.toHaveBeenCalled();
    expect(json).not.toHaveBeenCalled();
  });

  it("offers the original of a video instead of a preview", () => {
    const blob = vi.spyOn(v2, "fetchPdfBlob");
    const json = vi.spyOn(v2, "fetchDocumentJson");
    renderPane({
      ...pdfSource,
      source_type: "youtube",
      url: "https://www.youtube.com/watch?v=abc123def45",
      start_seconds: 754
    });
    expect(screen.getByText(/preview isn't available for this video yet/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open original ↗" })).toHaveAttribute(
      "href",
      "https://www.youtube.com/watch?v=abc123def45"
    );
    expect(blob).not.toHaveBeenCalled();
    expect(json).not.toHaveBeenCalled();
  });

  it("never links an original that isn't an http(s) address", () => {
    renderPane({ ...pdfSource, source_type: "image", url: "javascript:alert(1)" });
    expect(screen.getByText(/preview isn't available/)).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("shows no link when an image has no original address", () => {
    renderPane({ ...pdfSource, source_type: "image", url: null });
    expect(screen.getByText(/preview isn't available/)).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
