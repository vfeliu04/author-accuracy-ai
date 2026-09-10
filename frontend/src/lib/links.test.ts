import { describe, expect, it } from "vitest";
import {
  MAX_LINK_LENGTH,
  checkLink,
  linkHost,
  linkHostPath,
  safeHttpUrl,
  sourceName
} from "./links";

describe("checkLink", () => {
  it("accepts an http(s) link, trimmed and without its #fragment", () => {
    expect(checkLink("  https://example.org/report?id=2#section-3  ", [])).toEqual({
      link: "https://example.org/report?id=2"
    });
    expect(checkLink("http://example.org", [])).toEqual({ link: "http://example.org/" });
  });

  it("refuses every other scheme", () => {
    for (const input of [
      "ftp://example.org/file.pdf",
      "javascript:alert(1)",
      "file:///etc/passwd",
      "mailto:someone@example.org",
      "data:text/html,<b>hi</b>"
    ]) {
      expect(checkLink(input, [])).toEqual({
        error: "Only http:// and https:// links can be added."
      });
    }
  });

  it("refuses text that isn't a link", () => {
    for (const input of ["example.org/report", "not a link", "https://exa mple.org"]) {
      const result = checkLink(input, []);
      expect("error" in result && result.error).toMatch(/isn't a valid link/);
    }
  });

  it("refuses links longer than the limit", () => {
    const prefix = "https://example.org/";
    const atLimit = `${prefix}${"a".repeat(MAX_LINK_LENGTH - prefix.length)}`;
    expect(atLimit).toHaveLength(MAX_LINK_LENGTH);
    expect(checkLink(atLimit, [])).toEqual({ link: atLimit });
    const result = checkLink(`${atLimit}b`, []);
    expect("error" in result && result.error).toMatch(/2,048 characters/);
  });

  it("refuses a duplicate, including one that differs only by its #fragment or spacing", () => {
    const existing = ["https://example.org/report"];
    const duplicate = { error: "That link is already added." };
    expect(checkLink("https://example.org/report", existing)).toEqual(duplicate);
    expect(checkLink("https://example.org/report#top", existing)).toEqual(duplicate);
    expect(checkLink("  https://EXAMPLE.org/report  ", existing)).toEqual(duplicate);
    expect(checkLink("https://example.org/report?page=2", existing)).toEqual({
      link: "https://example.org/report?page=2"
    });
  });

  it("refuses YouTube links on every YouTube host", () => {
    for (const input of [
      "https://www.youtube.com/watch?v=abc123def45",
      "https://youtube.com/watch?v=abc123def45",
      "https://m.youtube.com/watch?v=abc123def45",
      "https://music.youtube.com/watch?v=abc123def45",
      "http://YouTube.com/shorts/abc",
      "https://youtu.be/abc123def45",
      "https://www.youtu.be/abc123def45",
      "https://www.youtube-nocookie.com/embed/abc123def45",
      "https://youtube.com./watch?v=abc"
    ]) {
      expect(checkLink(input, [])).toEqual({ error: "YouTube links aren't supported yet." });
    }
  });

  it("does not mistake look-alike hosts for YouTube", () => {
    for (const input of [
      "https://notyoutube.com/watch",
      "https://youtube.com.example.org/watch",
      "https://example.org/youtube.com/watch",
      "https://youtube.co/watch"
    ]) {
      expect(checkLink(input, [])).toHaveProperty("link");
    }
  });
});

describe("link display helpers", () => {
  it("shows the host, and the host plus a readable path", () => {
    expect(linkHost("https://www.example.org:8443/a/b?c=1#d")).toBe("www.example.org:8443");
    expect(linkHostPath("https://example.org/")).toBe("example.org");
    expect(linkHostPath("https://example.org/water/report%202024?id=7")).toBe(
      "example.org/water/report 2024"
    );
    expect(linkHostPath("https://example.org/bad%E0%A4%A")).toBe("example.org/bad%E0%A4%A");
  });

  it("falls back to the raw text when it isn't a link", () => {
    expect(linkHost("not a link")).toBe("not a link");
    expect(linkHostPath("not a link")).toBe("not a link");
  });
});

describe("sourceName", () => {
  it("prefers the title, then the link's host and path, then the fallback", () => {
    expect(sourceName({ title: "Water in Crisis", url: "https://example.org/w" }, "x")).toBe(
      "Water in Crisis"
    );
    expect(sourceName({ title: null, url: "https://example.org/w?id=1" }, "x")).toBe(
      "example.org/w"
    );
    expect(sourceName({ title: "", url: null }, "Source")).toBe("Source");
  });
});

describe("safeHttpUrl", () => {
  it("returns only http and https links", () => {
    expect(safeHttpUrl("https://example.org/a")).toBe("https://example.org/a");
    expect(safeHttpUrl("http://example.org")).toBe("http://example.org/");
    expect(safeHttpUrl("javascript:alert(1)")).toBeNull();
    expect(safeHttpUrl("JavaScript:alert(1)")).toBeNull();
    expect(safeHttpUrl("data:text/html,hi")).toBeNull();
    expect(safeHttpUrl("/relative/path")).toBeNull();
    expect(safeHttpUrl("")).toBeNull();
    expect(safeHttpUrl(null)).toBeNull();
    expect(safeHttpUrl(undefined)).toBeNull();
  });
});
