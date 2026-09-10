import { describe, expect, it } from "vitest";
import { humanizeError, linksNamedIn, namedLink } from "./errors";

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

describe("humanizeError reads the words around a link, never the link itself", () => {
  it("does not let a word in a link's address pick the message", () => {
    expect(humanizeError("Could not read https://example.org/billing/faq: HTTP 404")).toBe(
      "https://example.org/billing/faq — The site returned an error (404)."
    );
    expect(
      humanizeError(
        "Could not read http://192.168.1.10/billing: resolves to a private or reserved network address"
      )
    ).toBe(
      "http://192.168.1.10/billing — That link points to a private network address, so it can't be opened."
    );
    expect(
      humanizeError(
        "TimeoutError: Could not read https://docs.example.org/batch-processing: timed out after 20s"
      )
    ).toBe("https://docs.example.org/batch-processing — The site took too long to respond.");
    // A certificate failure is not a timeout, whatever the page is called.
    expect(
      humanizeError(
        "Could not read https://www.irs.gov/filing/individuals/when-to-file/deadlines: SSLError certificate verify failed"
      )
    ).toBeNull();
  });

  it("ignores the words in every link a message names, not only the first", () => {
    expect(
      humanizeError(
        "FetchError: Fetching 'https://example.org/moved' (redirected from 'https://example.org/billing') failed: the server answered HTTP 410"
      )
    ).toBe("https://example.org/moved — The site returned an error (410).");
  });

  it("names a link with parentheses or an apostrophe in full", () => {
    const wiki = "https://en.wikipedia.org/wiki/Mercury_(planet)";
    expect(humanizeError(`Could not read ${wiki}: HTTP 404`)).toBe(
      `${wiki} — The site returned an error (404).`
    );
    expect(
      humanizeError(`FetchError: Fetching '${wiki}' failed: the server answered HTTP 404`)
    ).toBe(`${wiki} — The site returned an error (404).`);
    expect(humanizeError(`Could not read "https://example.org/it's-here": HTTP 404`)).toBe(
      "https://example.org/it's-here — The site returned an error (404)."
    );
  });

  it("marks a link the message cut short", () => {
    const shown = `https://example.org/${"a".repeat(180)}`;
    expect(
      humanizeError(`FetchError: Fetching '${shown}...' failed: the server answered HTTP 404`)
    ).toBe(`${shown}… — The site returned an error (404).`);
  });
});

describe("namedLink", () => {
  it("keeps the parentheses and apostrophes that belong to a link", () => {
    expect(
      namedLink("Could not read https://en.wikipedia.org/wiki/Mercury_(planet): HTTP 404")
    ).toBe("https://en.wikipedia.org/wiki/Mercury_(planet)");
    expect(namedLink(`Could not read "https://example.org/it's-here": HTTP 404`)).toBe(
      "https://example.org/it's-here"
    );
    expect(namedLink("Fetching 'https://example.org/a_(b)_c' failed")).toBe(
      "https://example.org/a_(b)_c"
    );
  });

  it("leaves out the quotes, brackets, and punctuation around a link", () => {
    expect(namedLink("Could not read 'https://example.org/a'. HTTP 500")).toBe(
      "https://example.org/a"
    );
    expect(namedLink("no answer (https://api.crossref.org/works/10.1/x): HTTP 503")).toBe(
      "https://api.crossref.org/works/10.1/x"
    );
    expect(namedLink("Refusing to fetch [http://[fd00::1]/admin]")).toBe("http://[fd00::1]/admin");
    expect(namedLink("Connection error.")).toBeNull();
  });
});

describe("linksNamedIn", () => {
  const wiki = "https://en.wikipedia.org/wiki/Mercury_(planet)";
  const other = "https://example.org/b";

  it("finds an added link however the message quotes it", () => {
    expect(linksNamedIn(`FetchError: Could not read ${wiki}: HTTP 404`, [wiki, other])).toEqual([
      wiki
    ]);
    expect(
      linksNamedIn(`FetchError: Fetching '${wiki}' failed: the server answered HTTP 404`, [
        other,
        wiki
      ])
    ).toEqual([wiki]);
    const apostrophe = "https://example.org/it's-here";
    expect(
      linksNamedIn(`Fetching "${apostrophe}" failed: the server answered HTTP 404`, [apostrophe])
    ).toEqual([apostrophe]);
  });

  it("does not blame a link whose address only starts another added link's", () => {
    const report = "https://example.org/report";
    const yearly = "https://example.org/report-2024";
    expect(linksNamedIn(`Could not read ${yearly}: HTTP 404`, [report, yearly])).toEqual([yearly]);
    expect(linksNamedIn(`Could not read ${report}: HTTP 404`, [report, yearly])).toEqual([report]);
  });

  it("blames the link a redirect started from", () => {
    const added = "https://example.org/a";
    expect(
      linksNamedIn(
        `Fetching 'https://www.example.org/a/' (redirected from '${added}') failed: the server answered HTTP 404`,
        [added]
      )
    ).toEqual([added]);
  });

  it("recognizes a long link the message cut short", () => {
    const long = `https://example.org/${"a".repeat(300)}`;
    expect(
      linksNamedIn(`Fetching '${long.slice(0, 200)}...' failed: the server answered HTTP 404`, [
        other,
        long
      ])
    ).toEqual([long]);
  });

  it("names nothing when no added link appears", () => {
    expect(linksNamedIn("APIConnectionError: Connection error.", [wiki, other])).toEqual([]);
    expect(linksNamedIn(`Could not read ${wiki}: HTTP 404`, [])).toEqual([]);
  });
});
