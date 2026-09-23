import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReferenceScan, ScannedReference } from "../api/types";
import * as v2 from "../api/v2";
import { scannedReference } from "../test/fixtures";
import UploadDialog from "./UploadDialog";

function pdf(name: string, bytes = 100): File {
  return new File([new Uint8Array(bytes)], name, { type: "application/pdf" });
}

const noLimits: ReferenceScan["limits"] = {
  text_truncated: false,
  references_dropped: 0,
  possibly_incomplete: false
};

const emptyScan: ReferenceScan = {
  text_source: "none",
  lookup: { status: "ok", detail: null },
  limits: noLimits,
  references: []
};


function scanOf(
  references: ScannedReference[],
  lookup: ReferenceScan["lookup"] = { status: "ok", detail: null },
  limits: ReferenceScan["limits"] = noLimits
): ReferenceScan {
  return { text_source: "heading", lookup, limits, references };
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
      scanOf([scannedReference({ title: "Water scarcity in the Mediterranean basin" })])
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
        scannedReference({
          title: "Water scarcity in the Mediterranean basin",
          doi: "10.1000/abc",
          retrievability: "paywalled"
        }),
        scannedReference({
          title: "Drought hotspots of the twenty-first century",
          authors: ["Smith, J."],
          year: 2020,
          retrievability: "pdf",
          suggested_url: "https://europepmc.org/articles/PMC1?pdf=render"
        }),
        scannedReference({ entry: "Anon. (n.d.). A note without a title." })
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
        scannedReference({
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
        scannedReference({ title: "Work one", retrievability: "pdf", suggested_url: "https://a.org/one" }),
        scannedReference({ title: "Work two", retrievability: "landing", suggested_url: "https://b.org/two#sec" }),
        scannedReference({ title: "Work one again", retrievability: "pdf", suggested_url: "https://a.org/one#dup" }),
        scannedReference({ title: "Work four", retrievability: "paywalled" })
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
    // The duplicate's tick is not dropped in silence: the row goes, covered
    // by the first copy, and the note says why the count is one short.
    expect(
      screen.getByText("Added 1 of 2 — 1 couldn't be added: That link is already added.")
    ).toHaveClass("modal__count");
  });

  it("stops adding at the source limit and says how many went in", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf(
        ["one", "two", "three"].map((n) =>
          scannedReference({ title: `Work ${n}`, retrievability: "pdf", suggested_url: `https://a.org/${n}` })
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
    // The tick that did not fit still counts: Verify holds, with the limit's
    // own message, until it is unticked.
    expect(screen.getByText("At most 20 sources per verification.")).toBeInTheDocument();
    expect(verify()).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: /Work three/ }));
    expect(screen.queryByText("At most 20 sources per verification.")).not.toBeInTheDocument();
    expect(verify()).toBeEnabled();
  });

  it("offers an address printed in the entry unticked, labelled as never checked", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        scannedReference({
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
      scanOf([scannedReference({ title: "Work one" }), scannedReference({ title: "Work two" })], {
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
          scannedReference({ title: "Work one", retrievability: "pdf", suggested_url: "https://a.org/one" }),
          scannedReference({ title: "Work two" })
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
      scannedReference({ title: "A one", retrievability: "pdf", suggested_url: "https://a.org/one" }),
      scannedReference({ title: "A two", retrievability: "pdf", suggested_url: "https://a.org/two" })
    ]);
    const scanB = scanOf([
      scannedReference({ title: "B one", retrievability: "pdf", suggested_url: "https://b.org/one" }),
      scannedReference({ title: "B printed", url: "https://b.org/printed", suggested_url: "https://b.org/printed" })
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

  it("says, in one muted line, when the scan read or kept only part of the list", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([scannedReference({ title: "Work one" })], { status: "ok", detail: null }, {
        text_truncated: true,
        references_dropped: 12,
        possibly_incomplete: false
      })
    );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(1)).toBeInTheDocument());
    const line = screen.getByText("Read the first part of a long reference list — 12 entries not shown");
    expect(line).toHaveClass("modal__count");
    expect(line).not.toHaveClass("modal__error");
  });

  it("names one dropped entry alone, and says nothing when nothing was cut", async () => {
    const scan = vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([scannedReference({ title: "Work one" })], { status: "ok", detail: null }, {
        text_truncated: false,
        references_dropped: 1,
        possibly_incomplete: false
      })
    );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(1)).toBeInTheDocument());
    expect(screen.getByText("1 entry not shown")).toHaveClass("modal__count");
    expect(screen.queryByText(/first part of a long reference list/)).not.toBeInTheDocument();

    cleanup();
    scan.mockResolvedValue(scanOf([scannedReference({ title: "Work one" })]));
    renderDialog();
    await addReport("other.pdf");
    await waitFor(() => expect(heading(1)).toBeInTheDocument());
    expect(screen.queryByText(/not shown|reference list/)).not.toBeInTheDocument();
  });

  it("styles the Add note as a note, and drops it with the report it was about", async () => {
    const scan = vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf(
        ["one", "two", "three"].map((n) =>
          scannedReference({ title: `Work ${n}`, retrievability: "pdf", suggested_url: `https://a.org/${n}` })
        )
      )
    );
    renderDialog();
    const files = [pdf("report.pdf"), ...Array.from({ length: 18 }, (_, i) => pdf(`doc${i}.pdf`))];
    fireEvent.change(fileInput(), { target: { files } });
    await waitFor(() => expect(heading(3)).toBeInTheDocument());
    fireEvent.click(addLinks());
    const note = screen.getByText("Added 2 of 3 — at most 20 sources per verification.");
    expect(note).toHaveClass("modal__count");
    expect(note).not.toHaveClass("modal__error");

    fireEvent.click(screen.getByRole("button", { name: "Remove report.pdf" }));
    expect(screen.queryByText(/^Added \d+ of \d+/)).not.toBeInTheDocument();

    // Another report's scan: still no note, even though 20 sources remain.
    scan.mockResolvedValue(emptyScan);
    await addReport("other.pdf");
    expect(await screen.findByText("No reference list found in this report")).toBeInTheDocument();
    expect(screen.getByText("Sources (20)")).toBeInTheDocument();
    expect(screen.queryByText(/^Added \d+ of \d+/)).not.toBeInTheDocument();
  });

  it("names a row with no title by what it prints, and never by nothing", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        scannedReference({ entry: "", doi: "10.1000/blank" }),
        scannedReference({ entry: "", url: "https://c.org/blank", suggested_url: "https://c.org/blank" }),
        scannedReference({ entry: "" })
      ])
    );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(3)).toBeInTheDocument());
    const byDoi = screen.getByText("10.1000/blank").closest(".file-row");
    expect(byDoi).toHaveAttribute("title", "10.1000/blank");
    const byUrl = screen.getByRole("checkbox", { name: /c\.org\/blank/ }).closest(".file-row");
    expect(byUrl).toHaveAttribute("title", "https://c.org/blank");
    const bare = screen.getByText("(untitled entry)").closest(".file-row");
    expect(bare).toHaveAttribute("title", "(untitled entry)");
    for (const row of document.querySelectorAll(".file-row[title]")) {
      expect(row.getAttribute("title")).not.toBe("");
    }
  });

  it("lists a suggested address the dialog would refuse without a tick, saying why", async () => {
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        scannedReference({
          title: "A recorded webinar",
          url: "https://www.youtube.com/watch?v=abc",
          suggested_url: "https://www.youtube.com/watch?v=abc"
        }),
        scannedReference({
          title: "A record whose copy is a video",
          retrievability: "pdf",
          suggested_url: "https://youtu.be/abc"
        }),
        scannedReference({ title: "Work one", retrievability: "pdf", suggested_url: "https://a.org/one" })
      ])
    );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(3)).toBeInTheDocument());
    expect(screen.getAllByText("YouTube links aren't supported yet.")).toHaveLength(2);
    expect(screen.queryByRole("checkbox", { name: /webinar/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /video/ })).not.toBeInTheDocument();
    expect(screen.getByText("www.youtube.com")).toBeInTheDocument();
    expect(screen.getByText("youtu.be")).toBeInTheDocument();
    // The refusal replaces the copy tag: only Work one reads "free PDF".
    expect(screen.getAllByText("free PDF")).toHaveLength(1);
    expect(addLinks()).toHaveTextContent("Add 1 link");

    fireEvent.click(addLinks());
    expect(screen.getByText("Sources (1)")).toBeInTheDocument();
    expect(screen.queryByText(/^Added \d+ of \d+/)).not.toBeInTheDocument();
    expect(screen.getAllByText("YouTube links aren't supported yet.")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: /^Add \d+ links?$/ })).not.toBeInTheDocument();
    expect(heading(2)).toBeInTheDocument();
  });

  it("sends the ticked free copies with Verify, added first like a typed link", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf([
        scannedReference({ title: "Work one", retrievability: "pdf", suggested_url: "https://a.org/one" }),
        scannedReference({ title: "Work two", retrievability: "landing", suggested_url: "https://b.org/two" }),
        scannedReference({ title: "Work three", retrievability: "paywalled" })
      ])
    );
    renderDialog();
    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf"), pdf("s.pdf")] } });
    await waitFor(() => expect(heading(3)).toBeInTheDocument());
    expect(addLinks()).toHaveTextContent("Add 2 links");
    typeLink("https://c.org/typed");

    fireEvent.click(verify());
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    const [, sourcesArg, linksArg] = create.mock.calls[0];
    expect((sourcesArg as File[]).map((f) => f.name)).toEqual(["s.pdf"]);
    expect(linksArg).toEqual(["https://a.org/one", "https://b.org/two", "https://c.org/typed"]);
  });

  it("holds Verify, with the limit's message, while the ticked copies would pass the source limit", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf(
        ["one", "two", "three"].map((n) =>
          scannedReference({ title: `Work ${n}`, retrievability: "pdf", suggested_url: `https://a.org/${n}` })
        )
      )
    );
    renderDialog();
    const files = [pdf("report.pdf"), ...Array.from({ length: 18 }, (_, i) => pdf(`doc${i}.pdf`))];
    fireEvent.change(fileInput(), { target: { files } });
    await waitFor(() => expect(heading(3)).toBeInTheDocument());

    // 18 sources and 3 ticks are 21: the hold shows before any click, as it
    // does for a link past the limit, and nothing is added or sent.
    expect(screen.getByText("At most 20 sources per verification.")).toBeInTheDocument();
    expect(verify()).toBeDisabled();
    fireEvent.click(verify());
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(create).not.toHaveBeenCalled();
    expect(screen.getByText("Sources (18)")).toBeInTheDocument();
    expect(screen.queryByText(/^Added \d+ of \d+/)).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Work three/ })).toBeChecked();

    // Unticking the one that would not fit lifts the hold; Verify then adds
    // the two ticked copies first, as it does with a typed link.
    fireEvent.click(screen.getByRole("checkbox", { name: /Work three/ }));
    expect(screen.queryByText("At most 20 sources per verification.")).not.toBeInTheDocument();
    expect(verify()).toBeEnabled();
    fireEvent.click(verify());
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    expect(create.mock.calls[0][2]).toEqual(["https://a.org/one", "https://a.org/two"]);
  });

  it("lifts the hold when a source is removed to make room, and sends every tick at the limit", async () => {
    const create = vi.spyOn(v2, "createRun").mockResolvedValue({ run_id: "r", job_id: "j" });
    vi.spyOn(v2, "scanReferences").mockResolvedValue(
      scanOf(
        ["one", "two", "three"].map((n) =>
          scannedReference({ title: `Work ${n}`, retrievability: "pdf", suggested_url: `https://a.org/${n}` })
        )
      )
    );
    renderDialog();
    const files = [pdf("report.pdf"), ...Array.from({ length: 18 }, (_, i) => pdf(`doc${i}.pdf`))];
    fireEvent.change(fileInput(), { target: { files } });
    await waitFor(() => expect(heading(3)).toBeInTheDocument());
    expect(verify()).toBeDisabled();

    // 17 sources and 3 ticks are exactly 20, which the limit allows.
    fireEvent.click(screen.getByRole("button", { name: "Remove doc0.pdf" }));
    expect(screen.getByText("Sources (17)")).toBeInTheDocument();
    expect(screen.queryByText("At most 20 sources per verification.")).not.toBeInTheDocument();
    expect(verify()).toBeEnabled();
    fireEvent.click(verify());
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    const [, sourcesArg, linksArg] = create.mock.calls[0];
    expect((sourcesArg as File[]).length).toBe(17);
    expect(linksArg).toEqual(["https://a.org/one", "https://a.org/two", "https://a.org/three"]);
    expect(screen.queryByText(/^Added \d+ of \d+/)).not.toBeInTheDocument();
  });

  it("says the scan may have missed entries, with the same Try again, and clears it on a full answer", async () => {
    // The server's completeness guard could not get a full answer for a
    // part (limits.possibly_incomplete): the list is shown, with one muted
    // line and the control a failed scan gets, and asking again is the
    // same refetch — a second answer without the flag clears both.
    const scan = vi
      .spyOn(v2, "scanReferences")
      .mockResolvedValueOnce(
        scanOf([scannedReference({ title: "Work one" })], { status: "ok", detail: null }, {
          text_truncated: false,
          references_dropped: 0,
          possibly_incomplete: true
        })
      )
      .mockResolvedValue(
        scanOf([scannedReference({ title: "Work one" }), scannedReference({ title: "Work two" })])
      );
    renderDialog();
    await addReport();
    await waitFor(() => expect(heading(1)).toBeInTheDocument());
    const line = screen.getByText("The scan may have missed some entries");
    expect(line).toHaveClass("modal__count");
    expect(line).not.toHaveClass("modal__error");
    expect(screen.getByText("Work one")).toBeInTheDocument();
    expect(screen.queryByText(/not shown|reference list/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(heading(2)).toBeInTheDocument());
    expect(screen.getByText("Work two")).toBeInTheDocument();
    expect(screen.queryByText("The scan may have missed some entries")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
    expect(scan).toHaveBeenCalledTimes(2);
  });

  it("offers to try a failed scan again, and lists the works when it answers", async () => {
    const scan = vi
      .spyOn(v2, "scanReferences")
      .mockRejectedValueOnce(new Error("The model did not answer."))
      .mockResolvedValue(scanOf([scannedReference({ title: "Work one" })]));
    renderDialog();
    await addReport();
    expect(await screen.findByText("The model did not answer.")).toBeInTheDocument();
    expect(scan).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(heading(1)).toBeInTheDocument());
    expect(screen.getByText("Work one")).toBeInTheDocument();
    expect(screen.queryByText("The model did not answer.")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
    expect(scan).toHaveBeenCalledTimes(2);
    expect(scan.mock.calls[1][0].name).toBe("report.pdf");
  });
});
