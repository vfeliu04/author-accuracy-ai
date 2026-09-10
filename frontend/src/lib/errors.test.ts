import { describe, expect, it } from "vitest";
import { humanizeError } from "./errors";

describe("humanizeError", () => {
  it("returns null for no error or one it doesn't recognize", () => {
    expect(humanizeError(null)).toBeNull();
    expect(humanizeError("")).toBeNull();
    expect(humanizeError("ValueError: Extraction produced 0 claims for run 'r'")).toBeNull();
  });

  it("keeps the existing translations", () => {
    expect(humanizeError("APIConnectionError: Connection error.")).toMatch(/laptop sleep/);
    expect(
      humanizeError("TimeoutError: Batch msgbatch_01 still 'in_progress' after 3600s")
    ).toMatch(/reattaches to it/);
    expect(humanizeError("Your credit balance is too low to access the API")).toMatch(
      /out of credit/
    );
  });

  it("explains a link to a private network address and names the link", () => {
    const hint = humanizeError(
      "Could not read https://10.0.0.5/admin: resolves to a private or reserved network address"
    );
    expect(hint).toContain(
      "That link points to a private network address, so it can't be opened."
    );
    expect(hint).toContain("https://10.0.0.5/admin");
  });

  it("explains a link that isn't a web page or PDF", () => {
    const hint = humanizeError(
      "Could not read https://example.org/archive.zip: unsupported content type 'application/zip'"
    );
    expect(hint).toContain("That link isn't a web page or PDF.");
    expect(hint).toContain("https://example.org/archive.zip");
  });

  it("reports the status a site answered with", () => {
    const hint = humanizeError("Could not read https://example.org/gone: HTTP 404");
    expect(hint).toContain("The site returned an error (404).");
    expect(hint).toContain("https://example.org/gone");
  });

  it("explains a slow site for both timeout phrasings", () => {
    for (const error of [
      "Could not read https://slow.example.org/page: timed out after 20s",
      "Could not read https://slow.example.org/page: exceeded the 60s deadline"
    ]) {
      const hint = humanizeError(error);
      expect(hint).toContain("The site took too long to respond.");
      expect(hint).toContain("https://slow.example.org/page");
    }
  });

  it("explains a page with no readable text", () => {
    expect(
      humanizeError("Could not read https://app.example.org/: no readable article text")
    ).toContain(
      "That page has no readable text. Pages that need JavaScript to show their content can't be read."
    );
  });

  it("keeps the named link free of the punctuation around it", () => {
    const hint = humanizeError("Could not read 'https://example.org/a'. HTTP 500");
    expect(hint).toContain("https://example.org/a —");
    expect(hint).not.toContain("https://example.org/a'");
  });

  it("still explains a link failure whose message names no link", () => {
    expect(humanizeError("resolves to a private or reserved network address")).toBe(
      "That link points to a private network address, so it can't be opened."
    );
  });

  it("does not mistake a registry outage or a model timeout for a failing link", () => {
    const registry = humanizeError(
      "Crossref gave no answer after 3 attempts (https://api.crossref.org/works/10.1/x): HTTP 503"
    );
    expect(registry).not.toMatch(/The site returned an error/);
    expect(registry).toMatch(/registry/);
    expect(humanizeError("APITimeoutError: Request timed out.")).toBeNull();
    expect(humanizeError("RuntimeError: HTTP 502 from the embeddings service")).toBeNull();
  });
});
