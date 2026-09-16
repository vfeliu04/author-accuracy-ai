import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as v2 from "../api/v2";
import UploadDialog from "./UploadDialog";

function pdf(name: string, bytes = 100): File {
  return new File([new Uint8Array(bytes)], name, { type: "application/pdf" });
}

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

  it("adds a link, shows its host with the full link on hover, and removes it", () => {
    renderDialog();
    expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    typeLink("https://www.example.org/reports/water?id=7");
    clickAdd();

    const host = screen.getByText("www.example.org");
    expect(host).toHaveAttribute("title", "https://www.example.org/reports/water?id=7");
    expect(host).toHaveClass("file-row__host");
    expect(linkInput()).toHaveValue("");
    expect(screen.getByText("Sources (1)")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Remove https://www.example.org/reports/water?id=7" })
    );
    expect(screen.queryByText("www.example.org")).not.toBeInTheDocument();
    expect(screen.getByText("Sources (0)")).toBeInTheDocument();
  });

  it("adds a link with Enter", () => {
    renderDialog();
    typeLink("https://example.org/page");
    fireEvent.keyDown(linkInput(), { key: "Enter" });
    expect(screen.getByText("example.org")).toHaveAttribute("title", "https://example.org/page");
    expect(linkInput()).toHaveValue("");
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
    expect(screen.getAllByText("example.org")).toHaveLength(1);
  });

  it("rejects YouTube links", () => {
    renderDialog();
    typeLink("https://youtu.be/abc123def45");
    clickAdd();
    expect(screen.getByText("YouTube links aren't supported yet.")).toBeInTheDocument();
    expect(linkInput()).toHaveAttribute("aria-invalid", "true");
    expect(screen.queryByText("youtu.be")).not.toBeInTheDocument();
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
