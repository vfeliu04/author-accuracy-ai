import { afterEach, describe, expect, it, vi } from "vitest";
import type { PageSnapshot } from "./types";
import {
  UnreadablePageError,
  createRun,
  fetchDocumentJson,
  getReport,
  listRuns,
  parsePageSnapshot
} from "./v2";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

function pdf(name: string): File {
  return new File([new Uint8Array([1])], name, { type: "application/pdf" });
}

// A fresh Response per call — a shared one errors on the second body read.
function stubCreateResponses() {
  const fetchMock = vi
    .fn()
    .mockImplementation(() => Promise.resolve(jsonResponse({ run_id: "r", job_id: "j" })));
  vi.stubGlobal("fetch", fetchMock);
  return (call: number) => (fetchMock.mock.calls[call][1] as RequestInit).body as FormData;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("v2 fetchers", () => {
  it("listRuns unwraps the runs array", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ runs: [{ id: "r1", status: "DONE", created_at: "t", error: null }] })
    );
    vi.stubGlobal("fetch", fetchMock);

    const runs = await listRuns();
    expect(runs).toHaveLength(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/api\/runs$/);
    expect((init as RequestInit).headers).toBeInstanceOf(Headers);
  });

  it("createRun posts multipart with the report and sources", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ run_id: "r", job_id: "j" }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await createRun(pdf("r.pdf"), [pdf("s.pdf")], []);

    expect(result).toEqual({ run_id: "r", job_id: "j" });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/api\/runs$/);
    expect(init.method).toBe("POST");
    const form = init.body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect((form.get("report") as File).name).toBe("r.pdf");
    expect(form.getAll("sources")).toHaveLength(1);
    expect(form.getAll("source_urls")).toEqual([]);
    expect(form.get("title")).toBeNull(); // omitted when not provided
  });

  it("createRun sends one source_urls part per link, alongside the files", async () => {
    const formOf = stubCreateResponses();
    await createRun(pdf("r.pdf"), [pdf("s.pdf")], [
      "https://example.org/a",
      "https://example.org/b?page=2"
    ]);
    const form = formOf(0);
    expect(form.getAll("source_urls")).toEqual([
      "https://example.org/a",
      "https://example.org/b?page=2"
    ]);
    expect(form.getAll("sources")).toHaveLength(1);
  });

  it("createRun sends links without any source files", async () => {
    const formOf = stubCreateResponses();
    await createRun(pdf("r.pdf"), [], ["https://example.org/a"], "Links only");
    const form = formOf(0);
    expect(form.getAll("sources")).toHaveLength(0);
    expect(form.getAll("source_urls")).toEqual(["https://example.org/a"]);
    expect(form.get("title")).toBe("Links only");
  });

  it("createRun sends a trimmed title and omits blank ones", async () => {
    const formOf = stubCreateResponses();
    await createRun(pdf("r.pdf"), [pdf("s.pdf")], [], "  My Study  ");
    expect(formOf(0).get("title")).toBe("My Study");

    await createRun(pdf("r.pdf"), [pdf("s.pdf")], [], "   ");
    expect(formOf(1).get("title")).toBeNull();
  });

  it("throws the server error text on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("no such run", { status: 404 })));
    await expect(getReport("x")).rejects.toThrow("no such run");
  });

  it("surfaces the sentence in a JSON error body instead of the raw JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "YouTube links are not supported yet" }, 400))
    );
    await expect(createRun(pdf("r.pdf"), [], ["https://youtu.be/x"])).rejects.toThrow(
      /^YouTube links are not supported yet$/
    );
  });

  it("keeps the raw body when a JSON error carries no sentence", async () => {
    const body = { detail: [{ loc: ["body", "report"], msg: "Field required" }] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(body, 422)));
    await expect(getReport("x")).rejects.toThrow(JSON.stringify(body));
  });

  it("sends the X-API-Key header when configured", async () => {
    vi.stubEnv("VITE_API_KEY", "secret-key");
    vi.resetModules();
    const { listRuns: freshListRuns } = await import("./v2");
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ runs: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await freshListRuns();
    const headers = (fetchMock.mock.calls[0][1] as RequestInit).headers as Headers;
    expect(headers.get("X-API-Key")).toBe("secret-key");
  });

  it("fetchDocumentJson reads a stored page from the file endpoint with the API key", async () => {
    vi.stubEnv("VITE_API_KEY", "secret-key");
    vi.resetModules();
    const { fetchDocumentJson: freshFetch } = await import("./v2");
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ schema: 1 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(freshFetch("r1", "d1")).resolves.toEqual({ schema: 1 });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/api\/runs\/r1\/documents\/d1\/file$/);
    expect((init.headers as Headers).get("X-API-Key")).toBe("secret-key");
  });

  it("fetchDocumentJson raises the server's error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "No such document in this run" }, 404))
    );
    await expect(fetchDocumentJson("r1", "nope")).rejects.toThrow("No such document in this run");
  });

  it("fetchDocumentJson refuses a body that isn't a stored page, with a sentence for the reader", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("%PDF-1.4 binary", { status: 200 }))
    );
    const read = fetchDocumentJson("r1", "d1");
    await expect(read).rejects.toThrow(UnreadablePageError);
    await expect(read).rejects.toThrow(/^This page's saved text can't be read\.$/);
  });
});

const page: PageSnapshot = {
  schema: 1,
  document: {
    title: "Water in Crisis",
    sections: [{ title: "Findings", page: null, text: "Two billion people lack safe water." }]
  },
  provenance: {
    url: "https://example.org/water",
    final_url: "https://www.example.org/water",
    fetched_at: "2026-09-01T10:00:00Z",
    content_type: "text/html",
    title: "Water in Crisis",
    authors: ["A. Author"],
    publisher: "Example Institute",
    publication_date: "2024-03-01",
    doi: null,
    scholarly: false
  }
};

describe("parsePageSnapshot", () => {
  it("accepts a page in the current format", () => {
    expect(parsePageSnapshot(page)).toEqual(page);
  });

  it("refuses any other format with an error a reader can understand", () => {
    for (const schema of [2, 0, "1", undefined]) {
      expect(() => parsePageSnapshot({ ...page, schema })).toThrow(UnreadablePageError);
      expect(() => parsePageSnapshot({ ...page, schema })).toThrow(
        "This page was saved in a format this version of the app can't display."
      );
    }
    expect(() => parsePageSnapshot(null)).toThrow(UnreadablePageError);
    expect(() => parsePageSnapshot("<html></html>")).toThrow(UnreadablePageError);
  });

  it("refuses a page whose text or origin is missing", () => {
    expect(() => parsePageSnapshot({ ...page, document: { title: "x" } })).toThrow(/incomplete/);
    expect(() => parsePageSnapshot({ ...page, provenance: null })).toThrow(/incomplete/);
    expect(() =>
      parsePageSnapshot({
        ...page,
        document: { title: null, sections: [{ title: "A", page: null }] }
      })
    ).toThrow(/incomplete/);
  });

  it("drops display fields of the wrong kind instead of rendering them", () => {
    const parsed = parsePageSnapshot({
      ...page,
      document: { title: { html: "<b>" }, sections: [{ title: 7, page: "3", text: "Body" }] },
      provenance: { ...page.provenance, publisher: ["x"], final_url: 42, authors: ["A", 3] }
    });
    expect(parsed.document.title).toBeNull();
    expect(parsed.document.sections).toEqual([{ title: "", page: null, text: "Body" }]);
    expect(parsed.provenance.publisher).toBeNull();
    expect(parsed.provenance.final_url).toBe("");
    expect(parsed.provenance.authors).toEqual(["A"]);
  });

  it("keeps a transcript section's time locator", () => {
    const parsed = parsePageSnapshot({
      ...page,
      document: {
        title: null,
        sections: [{ title: "", page: null, text: "hi", start_seconds: 12, end_seconds: 30 }]
      }
    });
    expect(parsed.document.sections[0]).toEqual({
      title: "",
      page: null,
      text: "hi",
      start_seconds: 12,
      end_seconds: 30
    });
  });
});
