import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReferenceScan, ScannedReference } from "../api/types";
import * as v2 from "../api/v2";
import UploadDialog from "./UploadDialog";

function pdf(name: string, bytes = 100): File {
  return new File([new Uint8Array(bytes)], name, { type: "application/pdf" });
}

const emptyScan: ReferenceScan = {
  text_source: "none",
  lookup: { status: "ok", detail: null },
  references: []
};

function cited(over: Partial<ScannedReference> = {}): ScannedReference {
  return {
    title: null,
    authors: [],
    year: null,
    doi: null,
    url: null,
    entry: "An entry as printed.",
    retrievability: "unknown",
    suggested_url: null,
    ...over
  };
}

function scanOf(
  references: ScannedReference[],
  lookup: ReferenceScan["lookup"] = { status: "ok", detail: null }
): ReferenceScan {
  return { text_source: "heading", lookup, references };
}

// Every report picked in a test is scanned; one quiet answer keeps the tests
// that are not about the checklist from touching the network.
beforeEach(() => {
  vi.spyOn(v2, "scanReferences").mockResolvedValue(emptyScan);
});

function renderDialog(onClose = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <UploadDialog onClose={onClose} />
      </MemoryRouter>
    </QueryClientProvider>
  );
  return onClose;
}

function fileInput(): HTMLInputElement {
  const input = document.querySelector('input[type="file"]');
  if (!input) throw new Error("file input not rendered");
  return input as HTMLInputElement;
}

const verify = () => screen.getByRole("button", { name: /Verify report|Uploading/ });
const linkInput = () => screen.getByLabelText("Add a link") as HTMLInputElement;
const typeLink = (value: string) => fireEvent.change(linkInput(), { target: { value } });
const clickAdd = () => fireEvent.click(screen.getByRole("button", { name: "Add" }));

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("UploadDialog", () => {
  it("makes the first PDF the report and pre-fills the name from its stem", async () => {
    renderDialog();
    fireEvent.change(fileInput(), {
      target: { files: [pdf("Coastal_Brief.pdf"), pdf("source_one.pdf")] }
    });
    await waitFor(() => expect(screen.getByText("Coastal_Brief.pdf")).toBeInTheDocument());
    expect(screen.getByLabelText("Name")).toHaveValue("Coastal_Brief");
    expect(screen.getByText("source_one.pdf")).toBeInTheDocument();
    expect(screen.getByText(/2 files/)).toBeInTheDocument();
  });

  it("keeps a user-edited name when the report changes", async () => {
    renderDialog();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "My Study" } });
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());
    expect(screen.getByLabelText("Name")).toHaveValue("My Study");
  });

  it("rejects non-PDF files with an inline error", async () => {
    renderDialog();
    fireEvent.change(fileInput(), {
      target: { files: [new File([new Uint8Array(4)], "notes.txt", { type: "text/plain" })] }
    });
    await waitFor(() =>
      expect(screen.getByText(/Only PDF files can be verified/)).toBeInTheDocument()
    );
    expect(screen.queryByText("notes.txt")).not.toBeInTheDocument();
  });

  it("disables Verify until a report and at least one source exist, then submits with the title", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    renderDialog();
    expect(verify()).toBeDisabled();

    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());
    expect(verify()).toBeDisabled(); // report alone is not enough

    fireEvent.change(fileInput(), { target: { files: [pdf("src.pdf")] } });
    await waitFor(() => expect(screen.getByText("src.pdf")).toBeInTheDocument());
    expect(verify()).toBeEnabled();

    fireEvent.click(verify());
    await waitFor(() => expect(create).toHaveBeenCalled());
    const [reportArg, sourcesArg, linksArg, titleArg] = create.mock.calls[0];
    expect((reportArg as File).name).toBe("report.pdf");
    expect((sourcesArg as File[]).map((f) => f.name)).toEqual(["src.pdf"]);
    expect(linksArg).toEqual([]);
    expect(titleArg).toBe("report");
  });

  it("accepts files via drag and drop", async () => {
    renderDialog();
    const dropzone = screen.getByRole("button", { name: /Drop the report PDF/ });
    fireEvent.drop(dropzone, {
      dataTransfer: { files: [pdf("dropped.pdf")] }
    });
    await waitFor(() => expect(screen.getByText("dropped.pdf")).toBeInTheDocument());
  });

  it("adds a link, shows its host and path with the full link on hover, and removes it", () => {
    renderDialog();
    expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    typeLink("https://www.example.org/reports/water?id=7");
    clickAdd();

    const row = screen.getByText("www.example.org/reports/water");
    expect(row).toHaveAttribute("title", "https://www.example.org/reports/water?id=7");
    expect(row).toHaveClass("file-row__host");
    expect(linkInput()).toHaveValue("");
    expect(screen.getByText("Sources (1)")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Remove https://www.example.org/reports/water?id=7" })
    );
    expect(screen.queryByText("www.example.org/reports/water")).not.toBeInTheDocument();
    expect(screen.getByText("Sources (0)")).toBeInTheDocument();
  });

  it("adds a link with Enter", () => {
    renderDialog();
    typeLink("https://example.org/page");
    fireEvent.keyDown(linkInput(), { key: "Enter" });
    expect(screen.getByText("example.org/page")).toHaveAttribute(
      "title",
      "https://example.org/page"
    );
    expect(linkInput()).toHaveValue("");
  });

  it("tells apart two links from the same site", () => {
    renderDialog();
    typeLink("https://www.who.int/news-room/fact-sheets/detail/drinking-water");
    clickAdd();
    typeLink("https://www.who.int/news-room/fact-sheets/detail/sanitation");
    clickAdd();
    const rows = Array.from(document.querySelectorAll(".file-row__host"), (row) => row.textContent);
    expect(rows).toEqual([
      "www.who.int/news-room/fact-sheets/detail/drinking-water",
      "www.who.int/news-room/fact-sheets/detail/sanitation"
    ]);
  });

  it("rejects a duplicate link, including one that differs only by its #fragment", () => {
    renderDialog();
    typeLink("https://example.org/page");
    clickAdd();

    typeLink("https://example.org/page");
    clickAdd();
    expect(screen.getByText("That link is already added.")).toBeInTheDocument();
    // The rejected text stays in the box so it can be fixed.
    expect(linkInput()).toHaveValue("https://example.org/page");

    // Editing clears the message; the fragment-only variant is still a duplicate.
    typeLink("https://example.org/page#results");
    expect(screen.queryByText("That link is already added.")).not.toBeInTheDocument();
    fireEvent.keyDown(linkInput(), { key: "Enter" });
    expect(screen.getByText("That link is already added.")).toBeInTheDocument();
    expect(screen.getAllByText("example.org/page")).toHaveLength(1);
  });

  it("rejects YouTube links", () => {
    renderDialog();
    typeLink("https://youtu.be/abc123def45");
    clickAdd();
    expect(screen.getByText("YouTube links aren't supported yet.")).toBeInTheDocument();
    expect(linkInput()).toHaveAttribute("aria-invalid", "true");
    expect(document.querySelector(".file-row__host")).toBeNull();
    expect(screen.getByText("Sources (0)")).toBeInTheDocument();
  });

  it("rejects anything but a full http(s) link", () => {
    renderDialog();
    typeLink("ftp://example.org/file.pdf");
    clickAdd();
    expect(screen.getByText("Only http:// and https:// links can be added.")).toBeInTheDocument();

    typeLink("example.org/report");
    clickAdd();
    expect(screen.getByText(/isn't a valid link/)).toBeInTheDocument();
    expect(screen.getByText("Sources (0)")).toBeInTheDocument();
  });

  it("verifies a report against links alone, sending each as a source_urls part", async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify({ run_id: "r", job_id: "j" }), {
          status: 202,
          headers: { "Content-Type": "application/json" }
        })
      )
    );
    vi.stubGlobal("fetch", fetchMock);
    const onClose = renderDialog();

    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());
    expect(verify()).toBeDisabled();

    typeLink("https://example.org/a#intro");
    clickAdd();
    typeLink("https://example.org/b");
    fireEvent.keyDown(linkInput(), { key: "Enter" });
    expect(verify()).toBeEnabled();
    expect(screen.getByText(/1 file · 2 links/)).toBeInTheDocument();

    fireEvent.click(verify());
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const form = (fetchMock.mock.calls[0][1] as RequestInit).body as FormData;
    expect((form.get("report") as File).name).toBe("report.pdf");
    expect(form.getAll("sources")).toHaveLength(0);
    expect(form.getAll("source_urls")).toEqual(["https://example.org/a", "https://example.org/b"]);
  });

  it("sends a link typed into the box but not yet added", async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify({ run_id: "r", job_id: "j" }), {
          status: 202,
          headers: { "Content-Type": "application/json" }
        })
      )
    );
    vi.stubGlobal("fetch", fetchMock);
    const onClose = renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf"), pdf("s.pdf")] } });
    await waitFor(() => expect(screen.getByText("s.pdf")).toBeInTheDocument());

    typeLink("https://example.org/the-key-source");
    fireEvent.click(verify());
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const form = (fetchMock.mock.calls[0][1] as RequestInit).body as FormData;
    expect(form.getAll("source_urls")).toEqual(["https://example.org/the-key-source"]);
  });

  it("enables Verify for a report plus a link typed into the box but not yet added, and sends it", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    const onClose = renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());
    expect(verify()).toBeDisabled();

    typeLink("   ");
    expect(verify()).toBeDisabled(); // whitespace is not a source

    typeLink("https://example.org/the-only-source");
    expect(verify()).toBeEnabled();
    fireEvent.click(verify());
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const [, sourcesArg, linksArg] = create.mock.calls[0];
    expect(sourcesArg).toEqual([]);
    expect(linksArg).toEqual(["https://example.org/the-only-source"]);
  });

  it("holds Verify on the only source being a typed link that can't be added, and says why", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());

    typeLink("example.org/no-scheme");
    fireEvent.click(verify());
    expect(screen.getByText(/isn't a valid link/)).toBeInTheDocument();
    expect(linkInput()).toHaveValue("example.org/no-scheme");
    expect(create).not.toHaveBeenCalled();
  });

  it("holds Verify on a link in the box that can't be added, and says why", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    const onClose = renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf"), pdf("s.pdf")] } });
    await waitFor(() => expect(screen.getByText("s.pdf")).toBeInTheDocument());

    typeLink("https://youtu.be/abc123def45");
    fireEvent.click(verify());
    expect(screen.getByText("YouTube links aren't supported yet.")).toBeInTheDocument();
    expect(linkInput()).toHaveValue("https://youtu.be/abc123def45");
    expect(create).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("adds a link in the box that would pass the source limit, instead of sending anything", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    renderDialog();
    const files = Array.from({ length: 21 }, (_, index) => pdf(`doc${index}.pdf`));
    fireEvent.change(fileInput(), { target: { files } });
    await waitFor(() => expect(screen.getByText("Sources (20)")).toBeInTheDocument());

    typeLink("https://example.org/one-more");
    fireEvent.click(verify());
    expect(screen.getByText("Sources (21)")).toBeInTheDocument();
    expect(screen.getByText("At most 20 sources per verification.")).toBeInTheDocument();
    expect(create).not.toHaveBeenCalled();
  });

  it("counts links toward the source limit", async () => {
    renderDialog();
    const files = Array.from({ length: 21 }, (_, index) => pdf(`doc${index}.pdf`));
    fireEvent.change(fileInput(), { target: { files } });
    await waitFor(() => expect(screen.getByText("Sources (20)")).toBeInTheDocument());
    expect(verify()).toBeEnabled();

    typeLink("https://example.org/one-more");
    clickAdd();
    expect(screen.getByText("Sources (21)")).toBeInTheDocument();
    expect(screen.getByText("At most 20 sources per verification.")).toBeInTheDocument();
    expect(verify()).toBeDisabled();
  });

  it("shows the server's refusal of a link as a sentence", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "'https://example.org/a' was added twice" }), {
          status: 400,
          headers: { "Content-Type": "application/json" }
        })
      )
    );
    renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());
    typeLink("https://example.org/a");
    clickAdd();
    fireEvent.click(verify());
    expect(
      await screen.findByText("'https://example.org/a' was added twice")
    ).toBeInTheDocument();
  });

  it("turns the server's refusal of an unusable link into one sentence", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: "not a usable link: Source URL 'https://example.org/a' has an invalid host"
          }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        )
      )
    );
    renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());
    typeLink("https://example.org/a");
    clickAdd();
    fireEvent.click(verify());
    expect(
      await screen.findByText("One of the links isn't a valid web address. Check it and try again.")
    ).toBeInTheDocument();
    expect(screen.queryByText(/not a usable link/)).not.toBeInTheDocument();
  });

  it("refuses a link the server would refuse before anything is sent", () => {
    renderDialog();
    typeLink("https://user:pw@example.org/report");
    clickAdd();
    expect(
      screen.getByText("Links with a username or password can't be added.")
    ).toBeInTheDocument();
    expect(screen.getByText("Sources (0)")).toBeInTheDocument();
  });
});

describe("UploadDialog reference checklist", () => {
  const addReport = async (name = "report.pdf") => {
    fireEvent.change(fileInput(), { target: { files: [pdf(name)] } });
    await waitFor(() => expect(screen.getByText(name)).toBeInTheDocument());
  };
  const heading = (count: number) =>
    screen.getByText(`Cited by the report, not among your sources (${count})`);
  const addLinks = () => screen.getByRole("button", { name: /^Add \d+ links?$/ });
  const pending = () => new Promise<ReferenceScan>(() => {});

  it("scans the report once it is picked, with that file, and not before", async () => {
    const scan = vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([cited({ title: "Water scarcity in the Mediterranean basin" })])
    );
    renderDialog();
    expect(scan).not.toHaveBeenCalled();
    await addReport();
    expect(await screen.findByText("Water scarcity in the Mediterranean basin")).toBeInTheDocument();
    expect(scan).toHaveBeenCalledTimes(1);
    expect(scan.mock.calls[0][0].name).toBe("report.pdf");
    expect(scan.mock.calls[0][1]).toBeInstanceOf(AbortSignal);
  });

  it("says the scan is running, then that no reference list was found", async () => {
    let finish!: (value: ReferenceScan) => void;
    vi.spyOn(v2, "scanReferences").mockImplementation(
      () => new Promise<ReferenceScan>((resolve) => {
        finish = resolve;
      })
    );
    renderDialog();
    await addReport();
    expect(screen.getByText("Scanning the report's references…")).toBeInTheDocument();
    await act(async () => finish(emptyScan));
    expect(await screen.findByText("No reference list found in this report")).toBeInTheDocument();
    expect(screen.queryByText("Scanning the report's references…")).not.toBeInTheDocument();
  });

  it("aborts the scan in flight when the report is swapped, and scans the new one", async () => {
    const scan = vi.spyOn(v2, "scanReferences").mockImplementation(pending);
    renderDialog();
    await addReport("first.pdf");
    await waitFor(() => expect(scan).toHaveBeenCalledTimes(1));
    const first = scan.mock.calls[0][1] as AbortSignal;
    expect(first.aborted).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Remove first.pdf" }));
    await waitFor(() => expect(first.aborted).toBe(true));

    await addReport("second.pdf");
    await waitFor(() => expect(scan).toHaveBeenCalledTimes(2));
    expect(scan.mock.calls[1][0].name).toBe("second.pdf");
    expect((scan.mock.calls[1][1] as AbortSignal).aborted).toBe(false);
  });

  it("aborts the scan in flight when the dialog closes", async () => {
    const scan = vi.spyOn(v2, "scanReferences").mockImplementation(pending);
    renderDialog();
    await addReport();
    await waitFor(() => expect(scan).toHaveBeenCalledTimes(1));
    const signal = scan.mock.calls[0][1] as AbortSignal;
    expect(signal.aborted).toBe(false);
    cleanup();
    expect(signal.aborted).toBe(true);
  });

  it("lists only the cited works not among the sources, and updates as sources are added", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        cited({
          title: "Water scarcity in the Mediterranean basin",
          doi: "10.1000/abc",
          retrievability: "paywalled"
        }),
        cited({
          title: "Drought hotspots of the twenty-first century",
          authors: ["Smith, J."],
          year: 2020,
          retrievability: "pdf",
          suggested_url: "https://europepmc.org/articles/PMC1?pdf=render"
        }),
        cited({ entry: "Anon. (n.d.). A note without a title." })
      ])
    );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(3)).toBeInTheDocument());
    expect(screen.getByText("Water scarcity in the Mediterranean basin")).toBeInTheDocument();
    expect(screen.getByText("paywalled")).toBeInTheDocument();
    expect(screen.getByText("Anon. (n.d.). A note without a title.")).toBeInTheDocument();

    // A link carrying the DOI stands for the first work.
    typeLink("https://doi.org/10.1000/abc");
    clickAdd();
    expect(screen.queryByText("Water scarcity in the Mediterranean basin")).not.toBeInTheDocument();
    expect(heading(2)).toBeInTheDocument();

    // A PDF named after an author and the year stands for the second.
    fireEvent.change(fileInput(), { target: { files: [pdf("Smith_2020.pdf")] } });
    await waitFor(() =>
      expect(screen.queryByText("Drought hotspots of the twenty-first century")).not.toBeInTheDocument()
    );
    expect(screen.getByText("Anon. (n.d.). A note without a title.")).toBeInTheDocument();
    expect(heading(1)).toBeInTheDocument();
  });

  it("removes a row live when a PDF named after the cited title is added", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        cited({
          title: "Water scarcity in the Mediterranean basin",
          retrievability: "pdf",
          suggested_url: "https://europepmc.org/articles/PMC1?pdf=render"
        })
      ])
    );
    renderDialog();
    await addReport();
    const box = await screen.findByRole("checkbox", { name: /Water scarcity/ });
    expect(box).toBeChecked();
    expect(screen.getByText("free PDF")).toBeInTheDocument();

    fireEvent.change(fileInput(), {
      target: { files: [pdf("Water_scarcity_in_the_Mediterranean_basin.pdf")] }
    });
    await waitFor(() =>
      expect(screen.queryByRole("checkbox", { name: /Water scarcity/ })).not.toBeInTheDocument()
    );
    expect(heading(0)).toBeInTheDocument();
    expect(screen.getByText("Every cited work is among your sources.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Add \d+ links?$/ })).not.toBeInTheDocument();
  });

  it("adds the ticked free copies as links, skipping a duplicate, and shows them as links", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        cited({ title: "Work one", retrievability: "pdf", suggested_url: "https://a.org/one" }),
        cited({ title: "Work two", retrievability: "landing", suggested_url: "https://b.org/two#sec" }),
        cited({ title: "Work one again", retrievability: "pdf", suggested_url: "https://a.org/one#dup" }),
        cited({ title: "Work four", retrievability: "paywalled" })
      ])
    );
    renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf"), pdf("s.pdf")] } });
    await waitFor(() => expect(heading(4)).toBeInTheDocument());
    expect(addLinks()).toHaveTextContent("Add 3 links");
    expect(screen.getByText("free copy")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: /Work two/ }));
    expect(addLinks()).toHaveTextContent("Add 2 links");
    fireEvent.click(addLinks());

    const rows = Array.from(
      document.querySelectorAll(".file-row__name.file-row__host"),
      (row) => row.textContent
    );
    expect(rows).toEqual(["a.org/one"]);
    expect(screen.getByText("a.org/one")).toHaveAttribute("title", "https://a.org/one");
    expect(screen.getByText("Sources (2)")).toBeInTheDocument();
    expect(screen.queryByText("Work one")).not.toBeInTheDocument();
    expect(screen.queryByText("Work one again")).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Work two/ })).not.toBeChecked();
    expect(screen.getByText("Work four")).toBeInTheDocument();
    expect(heading(2)).toBeInTheDocument();
    expect(screen.queryByText(/^Added \d+ of \d+/)).not.toBeInTheDocument();
  });

  it("stops adding at the source limit and says how many went in", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf(
        ["one", "two", "three"].map((n) =>
          cited({ title: `Work ${n}`, retrievability: "pdf", suggested_url: `https://a.org/${n}` })
        )
      )
    );
    renderDialog();
    const files = [pdf("report.pdf"), ...Array.from({ length: 18 }, (_, i) => pdf(`doc${i}.pdf`))];
    fireEvent.change(fileInput(), { target: { files } });
    await waitFor(() => expect(screen.getByText("Sources (18)")).toBeInTheDocument());
    await waitFor(() => expect(heading(3)).toBeInTheDocument());

    fireEvent.click(addLinks());
    expect(screen.getByText("Added 2 of 3 — at most 20 sources per verification.")).toBeInTheDocument();
    expect(screen.getByText("Sources (20)")).toBeInTheDocument();
    expect(
      Array.from(document.querySelectorAll(".file-row__name.file-row__host"), (row) => row.textContent)
    ).toEqual(["a.org/one", "a.org/two"]);
    expect(screen.getByRole("checkbox", { name: /Work three/ })).toBeChecked();
    expect(heading(1)).toBeInTheDocument();
    expect(verify()).toBeEnabled();
  });

  it("offers an address printed in the entry unticked, labelled as never checked", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        cited({
          title: "A working paper on the web",
          url: "https://c.org/printed",
          suggested_url: "https://c.org/printed"
        })
      ])
    );
    renderDialog();
    await addReport();
    const box = await screen.findByRole("checkbox", { name: /A working paper/ });
    expect(box).not.toBeChecked();
    expect(screen.getByText("printed in the entry, not checked")).toBeInTheDocument();
    expect(addLinks()).toHaveTextContent("Add 0 links");
    expect(addLinks()).toBeDisabled();

    fireEvent.click(box);
    expect(addLinks()).toHaveTextContent("Add 1 link");
    fireEvent.click(addLinks());
    expect(screen.getByText("c.org/printed")).toHaveClass("file-row__host");
    expect(screen.queryByRole("checkbox", { name: /A working paper/ })).not.toBeInTheDocument();
    expect(verify()).toBeEnabled();
  });

  it("keeps Verify usable and shows the message when the scan fails", async () => {
    vi.spyOn(v2, "scanReferences").mockRejectedValue(new Error("“report.pdf” is not a PDF."));
    renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf"), pdf("s.pdf")] } });
    await waitFor(() => expect(screen.getByText("s.pdf")).toBeInTheDocument());
    expect(await screen.findByText("“report.pdf” is not a PDF.")).toBeInTheDocument();
    expect(screen.getByText("“report.pdf” is not a PDF.")).toHaveClass("modal__error");
    expect(verify()).toBeEnabled();
    expect(screen.queryByText(/Cited by the report/)).not.toBeInTheDocument();
  });

  it("tags every row when no lookup was configured", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([cited({ title: "Work one" }), cited({ title: "Work two" })], {
        status: "unconfigured",
        detail: null
      })
    );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(2)).toBeInTheDocument());
    expect(screen.getAllByText("lookup not set up")).toHaveLength(2);
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("tags the rows left unresolved when the lookup failed part-way, and says why", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf(
        [
          cited({ title: "Work one", retrievability: "pdf", suggested_url: "https://a.org/one" }),
          cited({ title: "Work two" })
        ],
        { status: "unavailable", detail: "Unpaywall answered 503 three times" }
      )
    );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(2)).toBeInTheDocument());
    expect(screen.getByText("free PDF")).toBeInTheDocument();
    expect(screen.getByText("lookup unavailable")).toBeInTheDocument();
    expect(
      screen.getByText("Free copies could not be looked up: Unpaywall answered 503 three times")
    ).toBeInTheDocument();
    expect(addLinks()).toHaveTextContent("Add 1 link");
  });

  it("forgets the ticks made for one report's scan when another report's scan arrives", async () => {
    const scanA = scanOf([
      cited({ title: "A one", retrievability: "pdf", suggested_url: "https://a.org/one" }),
      cited({ title: "A two", retrievability: "pdf", suggested_url: "https://a.org/two" })
    ]);
    const scanB = scanOf([
      cited({ title: "B one", retrievability: "pdf", suggested_url: "https://b.org/one" }),
      cited({ title: "B printed", url: "https://b.org/printed", suggested_url: "https://b.org/printed" })
    ]);
    vi.spyOn(v2, "scanReferences").mockImplementation((file) =>
      Promise.resolve(file.name === "a.pdf" ? scanA : scanB)
    );
    renderDialog();
    await addReport("a.pdf");
    // Untick row 0; row 1 stays ticked. Under a set not keyed to its scan,
    // B's row 0 (a free copy) would come up unticked and B's row 1 (an
    // address never checked) ticked: the shape of A's choices, not B's.
    fireEvent.click(await screen.findByRole("checkbox", { name: /A one/ }));
    expect(screen.getByRole("checkbox", { name: /A one/ })).not.toBeChecked();
    expect(addLinks()).toHaveTextContent("Add 1 link");

    fireEvent.click(screen.getByRole("button", { name: "Remove a.pdf" }));
    await addReport("b.pdf");
    expect(await screen.findByRole("checkbox", { name: /B one/ })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /B printed/ })).not.toBeChecked();
    expect(addLinks()).toHaveTextContent("Add 1 link");
  });
});
