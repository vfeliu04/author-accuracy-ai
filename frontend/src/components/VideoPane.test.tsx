import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PageSnapshot } from "../api/types";
import * as v2 from "../api/v2";
import { UnreadablePageError } from "../api/v2";
import { videoSnapshot } from "../test/fixtures";
import VideoPane from "./VideoPane";

const page: PageSnapshot = videoSnapshot();

function renderPane(props: Partial<ComponentProps<typeof VideoPane>> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <VideoPane runId="r" docId="d" startSeconds={null} section={null} quote={null} {...props} />
    </QueryClientProvider>
  );
}

afterEach(() => vi.restoreAllMocks());

describe("VideoPane", () => {
  it("embeds the video at the cited moment, and shows the cited caption window", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const { container } = renderPane({
      startSeconds: 75,
      section: "1:15–2:29",
      quote: "Two billion people lack safe water"
    });
    const frame = await screen.findByTitle("How Water Crises Start");
    expect(frame.tagName).toBe("IFRAME");
    expect(frame).toHaveAttribute(
      "src",
      "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?start=75"
    );
    expect(frame).toHaveAttribute("referrerPolicy", "strict-origin-when-cross-origin");
    expect(frame).toHaveAttribute("allowFullScreen");
    expect(frame).toHaveAttribute("loading", "lazy");
    expect(screen.getByRole("heading", { name: "1:15–2:29" })).toBeInTheDocument();
    const mark = container.querySelector("mark") as HTMLElement;
    expect(mark.textContent).toBe("Two billion people lack safe water");
    expect(screen.getByText("Example Channel")).toBeInTheDocument();
    expect(screen.getByText("2024-03-01")).toBeInTheDocument();
    expect(screen.getByText("Automatic captions (English)")).toBeInTheDocument();
    const open = screen.getByRole("link", { name: "Open on YouTube ↗" });
    expect(open).toHaveAttribute("href", "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=75s");
    expect(open).toHaveAttribute("target", "_blank");
    expect(open).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("reloads the player for another claim cited at the same moment", async () => {
    // Two claims citing one window share a start: the user may have played on
    // in between, and choosing the other claim must bring the player back.
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const pane = (claimId: string) => (
      <QueryClientProvider client={client}>
        <VideoPane runId="r" docId="d" claimId={claimId} startSeconds={75} section={null} quote={null} />
      </QueryClientProvider>
    );
    const { rerender } = render(pane("claim-a"));
    const first = await screen.findByTitle("How Water Crises Start");
    rerender(pane("claim-b"));
    const second = await screen.findByTitle("How Water Crises Start");
    expect(second).not.toBe(first);
    expect(second).toHaveAttribute("src", "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?start=75");
  });

  it("omits the embed's start param when the cited moment is 0", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    renderPane({ startSeconds: 0 });
    const frame = await screen.findByTitle("How Water Crises Start");
    expect(frame).toHaveAttribute("src", "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ");
  });

  it("omits the embed's start param when no moment was cited", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    renderPane({ startSeconds: null });
    const frame = await screen.findByTitle("How Water Crises Start");
    expect(frame).toHaveAttribute("src", "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ");
  });

  it("picks the window spanning the cited moment when no window starts exactly there", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    renderPane({ startSeconds: 100 });
    expect(await screen.findByRole("heading", { name: "1:15–2:29" })).toBeInTheDocument();
  });

  it("falls back to the section named by the citation when no start matches", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    renderPane({ startSeconds: null, section: "0:00–1:15" });
    expect(await screen.findByRole("heading", { name: "0:00–1:15" })).toBeInTheDocument();
  });

  it("shows the first window rather than nothing when no citation matches at all", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    renderPane({ startSeconds: 9999, section: "nowhere" });
    expect(await screen.findByRole("heading", { name: "0:00–1:15" })).toBeInTheDocument();
  });

  it("shows a note instead of a player when the channel disallows embedding", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      provenance: { ...page.provenance, video: { ...page.provenance.video!, embeddable: false } }
    });
    renderPane({});
    expect(
      await screen.findByText("This video can't be played here.")
    ).toBeInTheDocument();
    expect(document.querySelector("iframe")).toBeNull();
  });

  it("says plainly when captions were written rather than machine-read, in their language", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      provenance: {
        ...page.provenance,
        video: { ...page.provenance.video!, captions: { kind: "manual", language: "es" } }
      }
    });
    renderPane({});
    expect(await screen.findByText("Captions (Spanish)")).toBeInTheDocument();
  });

  it("falls back to the raw language tag when it can't be read", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      provenance: {
        ...page.provenance,
        video: { ...page.provenance.video!, captions: { kind: "manual", language: "xx-bogus-tag" } }
      }
    });
    renderPane({});
    expect(await screen.findByText(/Captions \(/)).toHaveTextContent("xx-bogus-tag");
  });

  it("shows a loading state, then the reader's own reason when a stored video can't be shown", async () => {
    // An UnreadablePageError never retries (unlike a generic Error, which
    // TanStack retries with backoff), so the pane settles at once.
    vi.spyOn(v2, "fetchDocumentJson").mockRejectedValue(
      new UnreadablePageError("This video's saved data can't be read.")
    );
    renderPane({});
    expect(screen.getByText("Loading video…")).toBeInTheDocument();
    expect(await screen.findByText("This video's saved data can't be read.")).toBeInTheDocument();
  });

  it("says so when a stored video's snapshot carries no video declarations", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      provenance: { ...page.provenance, video: undefined }
    });
    renderPane({});
    expect(await screen.findByText(/Could not load this video/)).toBeInTheDocument();
  });
});
