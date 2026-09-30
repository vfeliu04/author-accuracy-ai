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
      .spyOn(v2, "fetchDocumentBlob")
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
    const blob = vi.spyOn(v2, "fetchDocumentBlob");
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

  it("shows an image source whole, saying what its quotes were checked against", async () => {
    const blob = vi
      .spyOn(v2, "fetchDocumentBlob")
      .mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    const json = vi.spyOn(v2, "fetchDocumentJson");
    renderPane({ ...pdfSource, source_type: "image", title: "stunting chart" });
    const shown = await screen.findByRole("img", { name: "Source image: stunting chart" });
    expect(shown).toHaveAttribute("src", "blob:fake");
    expect(
      screen.getByText(
        "Quotes from this image are checked against the model's reading of it, not against the image itself."
      )
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open full size ↗" })).toHaveAttribute(
      "href",
      "blob:fake"
    );
    expect(blob).toHaveBeenCalledWith("r", "d");
    expect(json).not.toHaveBeenCalled();
    expect(screen.queryByTitle("source")).not.toBeInTheDocument(); // no PDF viewer
  });

  it("says so when an image source cannot be loaded", async () => {
    vi.spyOn(v2, "fetchDocumentBlob").mockRejectedValue(new Error("Document file is unavailable"));
    renderPane({ ...pdfSource, source_type: "image" });
    expect(
      await screen.findByText("Could not load the image: Document file is unavailable")
    ).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("shows a YouTube video source as its player, with start_seconds and section threaded from the source", async () => {
    const videoPage: PageSnapshot = {
      schema: 1,
      document: {
        title: "How Water Crises Start",
        sections: [
          {
            title: "12:34–13:49",
            page: null,
            text: "Two billion people lack safe water.",
            start_seconds: 754,
            end_seconds: 829
          }
        ]
      },
      provenance: {
        url: "https://www.youtube.com/watch?v=abc123def45",
        final_url: "https://www.youtube.com/watch?v=abc123def45",
        fetched_at: "2026-09-01T10:00:00Z",
        title: "How Water Crises Start",
        authors: [],
        publisher: "Example Channel",
        publication_date: null,
        doi: null,
        scholarly: false,
        video: {
          id: "abc123def45",
          channel_id: "UCabc",
          channel_verified: true,
          duration_seconds: 900,
          embeddable: true,
          captions: { kind: "automatic", language: "en" }
        }
      }
    };
    const json = vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(videoPage);
    const blob = vi.spyOn(v2, "fetchDocumentBlob");
    renderPane(
      {
        ...pdfSource,
        source_type: "youtube",
        url: "https://www.youtube.com/watch?v=abc123def45",
        section: "12:34–13:49",
        start_seconds: 754
      },
      "Two billion people lack safe water"
    );
    const frame = await screen.findByTitle("How Water Crises Start");
    expect(frame).toHaveAttribute(
      "src",
      "https://www.youtube-nocookie.com/embed/abc123def45?start=754"
    );
    expect(document.querySelector("mark")?.textContent).toBe("Two billion people lack safe water");
    expect(json).toHaveBeenCalledWith("r", "d");
    expect(blob).not.toHaveBeenCalled();
    expect(screen.queryByTitle("source")).not.toBeInTheDocument();
  });

  it("offers the original of a source type this build doesn't know, instead of a preview", () => {
    const blob = vi.spyOn(v2, "fetchDocumentBlob");
    const json = vi.spyOn(v2, "fetchDocumentJson");
    renderPane({
      ...pdfSource,
      source_type: "podcast" as EvidenceSource["source_type"],
      url: "https://example.org/episode"
    });
    expect(screen.getByText(/preview isn't available for this source yet/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open original ↗" })).toHaveAttribute(
      "href",
      "https://example.org/episode"
    );
    expect(blob).not.toHaveBeenCalled();
    expect(json).not.toHaveBeenCalled();
  });

  it("never links an original that isn't an http(s) address", () => {
    renderPane({
      ...pdfSource,
      source_type: "podcast" as EvidenceSource["source_type"],
      url: "javascript:alert(1)"
    });
    expect(screen.getByText(/preview isn't available/)).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("shows no link when a source of an unknown type has no original address", () => {
    renderPane({ ...pdfSource, source_type: "podcast" as EvidenceSource["source_type"], url: null });
    expect(screen.getByText(/preview isn't available/)).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
