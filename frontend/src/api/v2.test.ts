import { afterEach, describe, expect, it, vi } from "vitest";
import { createRun, getReport, listRuns } from "./v2";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" }
  });
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

    const report = new File([new Uint8Array([1])], "r.pdf", { type: "application/pdf" });
    const source = new File([new Uint8Array([2])], "s.pdf", { type: "application/pdf" });
    const result = await createRun(report, [source]);

    expect(result).toEqual({ run_id: "r", job_id: "j" });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/api\/runs$/);
    expect(init.method).toBe("POST");
    const form = init.body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect((form.get("report") as File).name).toBe("r.pdf");
    expect(form.getAll("sources")).toHaveLength(1);
  });

  it("throws the server error text on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("no such run", { status: 404 })));
    await expect(getReport("x")).rejects.toThrow("no such run");
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
});
