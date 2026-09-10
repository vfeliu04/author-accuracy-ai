import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PageSnapshot } from "../api/types";
import * as v2 from "../api/v2";
import ReadablePane from "./ReadablePane";

const page: PageSnapshot = {
  schema: 1,
  document: {
    title: "Water in Crisis",
    sections: [
      { title: "", page: null, text: "Lead paragraph about water." },
      {
        title: "Findings",
        page: null,
        text:
          "Two billion people lack safe water.\n\nDemand rose 20–30% since 2000.\n" +
          "| Region | Share |\n| --- | --- |\n| Asia | 60% |"
      },
      { title: "Methods", page: null, text: "Surveys ran in 2023." }
    ]
  },
  provenance: {
    url: "https://example.org/water",
    final_url: "https://www.example.org/water",
    fetched_at: "2026-09-01T10:00:00Z",
    content_type: "text/html",
    title: "Water in Crisis",
    authors: [],
    publisher: "Example Institute",
    publication_date: "2024-03-01",
    doi: null,
    scholarly: false
  }
};

function renderPane(props: Partial<ComponentProps<typeof ReadablePane>> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReadablePane
        runId="r"
        docId="d"
        url="https://example.org/water"
        quote={null}
        section={null}
        {...props}
      />
    </QueryClientProvider>
  );
}

// jsdom has no scrolling; record which elements were asked to scroll.
const originalScrollIntoView = Element.prototype.scrollIntoView;
let scrolled: Element[] = [];

beforeEach(() => {
  scrolled = [];
  Element.prototype.scrollIntoView = vi.fn(function (this: Element) {
    scrolled.push(this);
  });
});

afterEach(() => {
  Element.prototype.scrollIntoView = originalScrollIntoView;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ReadablePane", () => {
  it("shows the page's title, publisher, date, and a safe link to the original", async () => {
    const read = vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    renderPane();
    expect(await screen.findByRole("heading", { name: "Water in Crisis" })).toBeInTheDocument();
    expect(read).toHaveBeenCalledWith("r", "d");
    expect(screen.getByText("Example Institute")).toBeInTheDocument();
    expect(screen.getByText("2024-03-01")).toBeInTheDocument();
    const open = screen.getByRole("link", { name: "Open original ↗" });
    expect(open).toHaveAttribute("href", "https://www.example.org/water");
    expect(open).toHaveAttribute("target", "_blank");
    expect(open).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("links the source's own address when the page records no final one", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      provenance: { ...page.provenance, final_url: "", publisher: null, publication_date: null }
    });
    renderPane({ url: "https://example.org/water?ref=1" });
    const open = await screen.findByRole("link", { name: "Open original ↗" });
    expect(open).toHaveAttribute("href", "https://example.org/water?ref=1");
    expect(screen.queryByText("Example Institute")).not.toBeInTheDocument();
  });

  it("links the address the page recorded when nothing else names one", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      provenance: { ...page.provenance, final_url: "" }
    });
    renderPane({ url: null });
    const open = await screen.findByRole("link", { name: "Open original ↗" });
    expect(open).toHaveAttribute("href", "https://example.org/water");
  });

  it("renders sections as headings and paragraphs, and table rows in their own block", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    renderPane();
    expect(await screen.findByRole("heading", { name: "Findings" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Methods" })).toBeInTheDocument();
    // The page title plus two titled sections — no empty heading for the lead.
    expect(screen.getAllByRole("heading")).toHaveLength(3);
    expect(screen.getByText("Lead paragraph about water.").tagName).toBe("P");
    expect(screen.getByText("Two billion people lack safe water.").tagName).toBe("P");
    expect(screen.getByText("Demand rose 20–30% since 2000.").tagName).toBe("P");
    const table = screen.getByText(/\| Region \| Share \|/);
    expect(table).toHaveClass("readable__table");
    expect(table.textContent).toBe("| Region | Share |\n| --- | --- |\n| Asia | 60% |");
  });

  it("marks the quote and scrolls it into view", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const { container } = renderPane({
      quote: "demand rose 20-30% since 2000",
      section: "Findings"
    });
    await waitFor(() => expect(container.querySelectorAll("mark")).toHaveLength(1));
    const mark = container.querySelector("mark") as HTMLElement;
    expect(mark.textContent).toBe("Demand rose 20–30% since 2000");
    await waitFor(() => expect(scrolled).toEqual([mark]));
  });

  it("splits the mark across the paragraphs a quote spans", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const { container } = renderPane({
      quote: "lack safe water. Demand rose",
      section: "Findings"
    });
    await waitFor(() => expect(container.querySelectorAll("mark")).toHaveLength(2));
    const marks = Array.from(container.querySelectorAll("mark"));
    expect(marks.map((mark) => mark.textContent)).toEqual(["lack safe water.", "Demand rose"]);
    expect(marks[0].closest("p")).not.toBe(marks[1].closest("p"));
    await waitFor(() => expect(scrolled).toEqual([marks[0]]));
  });

  it("marks a quote inside a table block", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const { container } = renderPane({ quote: "| Asia | 60% |", section: "Findings" });
    await waitFor(() => expect(container.querySelectorAll("mark")).toHaveLength(1));
    const mark = container.querySelector("mark") as HTMLElement;
    expect(mark.textContent).toBe("| Asia | 60% |");
    expect(mark.closest(".readable__table")).not.toBeNull();
  });

  it("prefers the cited section when the same words appear earlier", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      document: {
        title: "Repeats",
        sections: [
          { title: "Intro", page: null, text: "Water use rose." },
          { title: "Data", page: null, text: "Water use rose." }
        ]
      }
    });
    const { container } = renderPane({ quote: "Water use rose", section: "Data" });
    await waitFor(() => expect(container.querySelectorAll("mark")).toHaveLength(1));
    const section = (container.querySelector("mark") as HTMLElement).closest("section");
    expect(section).toContainElement(screen.getByRole("heading", { name: "Data" }));
  });

  it("finds a quote that sits outside the cited section", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const { container } = renderPane({ quote: "Surveys ran in 2023", section: "Findings" });
    await waitFor(() => expect(container.querySelectorAll("mark")).toHaveLength(1));
    const section = (container.querySelector("mark") as HTMLElement).closest("section");
    expect(section).toContainElement(screen.getByRole("heading", { name: "Methods" }));
  });

  it("finds a quote cited without a section name, wherever it sits", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const { container } = renderPane({ quote: "two billion people lack", section: null });
    await waitFor(() => expect(container.querySelectorAll("mark")).toHaveLength(1));
    const section = (container.querySelector("mark") as HTMLElement).closest("section");
    expect(section).toContainElement(screen.getByRole("heading", { name: "Findings" }));
  });

  it("prefers an untitled section for a quote cited without a section name", async () => {
    // An untitled section's quotes arrive with no section name at all.
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      document: {
        title: "Repeats",
        sections: [
          { title: "Intro", page: null, text: "Water use rose." },
          { title: "", page: null, text: "Water use rose." }
        ]
      }
    });
    const { container } = renderPane({ quote: "Water use rose", section: null });
    await waitFor(() => expect(container.querySelectorAll("mark")).toHaveLength(1));
    const section = (container.querySelector("mark") as HTMLElement).closest("section");
    expect(section).toHaveAttribute("data-section", "1");
  });

  it("scrolls to the cited section when the quote isn't on the page", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue(page);
    const { container } = renderPane({
      quote: "a sentence this page never says",
      section: "Methods"
    });
    const heading = await screen.findByRole("heading", { name: "Methods" });
    await waitFor(() => expect(scrolled).toHaveLength(1));
    expect(scrolled[0]).toContainElement(heading);
    expect(container.querySelector("mark")).toBeNull();
  });

  it("renders page text literally, never as markup", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      document: {
        title: "<b>Bold</b>",
        sections: [
          {
            title: "<img src=x onerror=alert(1)>",
            page: null,
            text: "<script>alert(1)</script>"
          }
        ]
      }
    });
    const { container } = renderPane();
    expect(await screen.findByText("<script>alert(1)</script>")).toBeInTheDocument();
    expect(screen.getByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
    expect(screen.getByText("<b>Bold</b>")).toBeInTheDocument();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
  });

  it("shows an original address that isn't http(s) as plain text, never as a link", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({
      ...page,
      provenance: {
        ...page.provenance,
        url: "javascript:alert(1)",
        final_url: "javascript:alert(1)"
      }
    });
    renderPane({ url: "javascript:alert(1)" });
    const origin = await screen.findByText("javascript:alert(1)");
    expect(origin.closest("a")).toBeNull();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("says at once, in plain words, when a stored page can't be read", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockImplementation(() => Promise.resolve(new Response("%PDF-1.4 binary", { status: 200 })))
    );
    renderPane();
    expect(await screen.findByText("This page's saved text can't be read.")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/JSON|Unexpected token|Could not load/);
  });

  it("refuses, visibly, a page stored in a format it can't read", async () => {
    vi.spyOn(v2, "fetchDocumentJson").mockResolvedValue({ ...page, schema: 2 });
    renderPane();
    expect(
      await screen.findByText(/saved in a format this version of the app can't display/)
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Water in Crisis" })).not.toBeInTheDocument();
  });
});
