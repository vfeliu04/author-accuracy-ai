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
    expect(checkLink(`${atLimit}b`, [])).toEqual({ error: "That link is too long to add." });
  });

  it("measures a link as it will be sent, and never quotes a length the typed text doesn't reach", () => {
    // 1,020 characters as typed; each "é" is sent as "%C3%A9".
    const typed = `https://example.org/${"é".repeat(1000)}`;
    expect(typed.length).toBeLessThan(MAX_LINK_LENGTH);
    expect(checkLink(typed, [])).toEqual({ error: "That link is too long to add." });
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
      "https://youtube.com./watch?v=abc",
      "https://youtube.com../watch?v=abc",
      "https://www.youtube.com.../watch?v=abc",
      "https://youtu.be../abc"
    ]) {
      expect(checkLink(input, [])).toEqual({ error: "YouTube links aren't supported yet." });
    }
  });

  it("refuses a link with a username or password in it", () => {
    for (const input of ["https://user@example.org/a", "https://user:pw@example.org/report"]) {
      expect(checkLink(input, [])).toEqual({
        error: "Links with a username or password can't be added."
      });
    }
  });

  it("refuses port 0 and a site name with characters no site name can have", () => {
    expect(checkLink("https://example.org:0/report", [])).toEqual({
      error: "That link has an invalid port."
    });
    for (const input of [
      "https://ex!ample.org/a",
      "https://exa$mple.org/a",
      "https://ex*ample.org/a"
    ]) {
      expect(checkLink(input, [])).toEqual({ error: "That link's site name isn't valid." });
    }
  });

  it("still accepts the site names and ports the server accepts", () => {
    for (const input of [
      "https://my_host.example.org/a",
      "https://example.org:8443/a",
      "http://[2001:db8::1]/a",
      "http://192.0.2.10/a",
      "https://bücher.de/a",
      "https://example.org../a"
    ]) {
      expect(checkLink(input, [])).toHaveProperty("link");
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

  it("shows an international host as it is written", () => {
    // Compared and sent in one spelling; shown the way people read it.
    expect(checkLink("https://bücher.de/katalog", [])).toEqual({
      link: "https://xn--bcher-kva.de/katalog"
    });
    expect(linkHost("https://xn--bcher-kva.de/katalog")).toBe("bücher.de");
    expect(linkHostPath("https://xn--mnchen-3ya.de:8080/stadt")).toBe("münchen.de:8080/stadt");
    expect(linkHost("https://www.xn--wgv71a119e.jp/")).toBe("www.日本語.jp");
    expect(sourceName({ title: null, url: "https://xn--fiqs8s.cn/news" }, "x")).toBe("中国.cn/news");
  });

  it("keeps a host that mixes Latin with look-alike letters in its encoded form", () => {
    // "аpple.com" with a Cyrillic "а" must not pass for apple.com.
    expect(linkHost("https://аpple.com/")).toBe("xn--pple-43d.com");
    expect(linkHostPath("https://xn--pple-43d.com/login")).toBe("xn--pple-43d.com/login");
  });

  it("keeps a host spelled wholly in look-alike letters in its encoded form", () => {
    // "аррӏе.com" and "еріс.com" are all Cyrillic, and read as apple.com and epic.com.
    expect(linkHost("https://xn--80ak6aa92e.com/")).toBe("xn--80ak6aa92e.com");
    expect(linkHostPath("https://www.xn--80ak6aa92e.com/login")).toBe(
      "www.xn--80ak6aa92e.com/login"
    );
    expect(linkHost("https://аррӏе.com/")).toBe("xn--80ak6aa92e.com");
    expect(linkHost("https://xn--e1awd7f.com/")).toBe("xn--e1awd7f.com");
    expect(linkHostPath("https://xn--e1awd7f.com/x")).toBe("xn--e1awd7f.com/x");
    // Greek, and a symbol that is no letter at all.
    expect(linkHost("https://xn--hxakic4aa.gr/")).toBe("xn--hxakic4aa.gr");
    expect(linkHost("https://xn--i-7iq.ws/")).toBe("xn--i-7iq.ws");
  });

  it("still shows hosts in scripts with no Latin look-alikes as they are written", () => {
    expect(linkHost("https://xn--bcher-2024-9db.de/")).toBe("bücher-2024.de");
    expect(linkHost("https://xn--3e0b707e.kr/")).toBe("한국.kr");
    expect(linkHost("https://xn--tck2c4fb.jp/")).toBe("コーヒー.jp");
    expect(linkHost("https://xn--mgbh0fb.eg/")).toBe("مثال.eg");
    expect(linkHost("https://xn--9dbne9b.co.il/")).toBe("שלום.co.il");
    expect(linkHost("https://xn--o3cw4h.th/")).toBe("ไทย.th");
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
