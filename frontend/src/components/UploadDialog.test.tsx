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

afterEach(() => vi.restoreAllMocks());

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
    const verify = () => screen.getByRole("button", { name: /Verify report|Uploading/ });
    expect(verify()).toBeDisabled();

    fireEvent.change(fileInput(), { target: { files: [pdf("report.pdf")] } });
    await waitFor(() => expect(screen.getByText("report.pdf")).toBeInTheDocument());
    expect(verify()).toBeDisabled(); // report alone is not enough

    fireEvent.change(fileInput(), { target: { files: [pdf("src.pdf")] } });
    await waitFor(() => expect(screen.getByText("src.pdf")).toBeInTheDocument());
    expect(verify()).toBeEnabled();

    fireEvent.click(verify());
    await waitFor(() => expect(create).toHaveBeenCalled());
    const [reportArg, sourcesArg, titleArg] = create.mock.calls[0];
    expect((reportArg as File).name).toBe("report.pdf");
    expect((sourcesArg as File[]).map((f) => f.name)).toEqual(["src.pdf"]);
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
});
