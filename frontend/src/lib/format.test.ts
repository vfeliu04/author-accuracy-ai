import { describe, expect, it } from "vitest";
import { citeLabel, formatTimestamp } from "./format";

describe("formatTimestamp", () => {
  it("labels time the way a video player does", () => {
    expect(formatTimestamp(0)).toBe("0:00");
    expect(formatTimestamp(7)).toBe("0:07");
    expect(formatTimestamp(754)).toBe("12:34");
    expect(formatTimestamp(3599.9)).toBe("59:59");
  });

  it("adds hours once the time passes the hour", () => {
    expect(formatTimestamp(3600)).toBe("1:00:00");
    expect(formatTimestamp(3723)).toBe("1:02:03");
    expect(formatTimestamp(36061)).toBe("10:01:01");
  });
});

const none = { page: null, section: null, start_seconds: null };

describe("citeLabel", () => {
  it("cites a PDF by page and omits a missing page", () => {
    expect(citeLabel({ ...none, source_type: "pdf", page: 3 })).toBe("p.3");
    expect(citeLabel({ ...none, source_type: "pdf" })).toBeNull();
  });

  it("never labels a page-less PDF item by its section", () => {
    // Docling leaves page unset for items without provenance; the PDF's
    // heading must not make it read like a web section.
    expect(citeLabel({ ...none, source_type: "pdf", section: "Methods" })).toBeNull();
  });

  it("cites a web page by section, never by page", () => {
    expect(citeLabel({ ...none, source_type: "web", section: "Key findings" })).toBe(
      "§ Key findings"
    );
    expect(citeLabel({ ...none, source_type: "web", page: 2 })).toBeNull();
    expect(citeLabel({ ...none, source_type: "web", section: "" })).toBeNull();
  });

  it("cites a video by start time, with hours past the hour", () => {
    expect(citeLabel({ ...none, source_type: "youtube", start_seconds: 0 })).toBe("0:00");
    expect(citeLabel({ ...none, source_type: "youtube", start_seconds: 754 })).toBe("12:34");
    expect(citeLabel({ ...none, source_type: "youtube", start_seconds: 3723 })).toBe("1:02:03");
    expect(citeLabel({ ...none, source_type: "youtube", page: 4 })).toBeNull();
  });

  it("cites an image as an image", () => {
    expect(citeLabel({ ...none, source_type: "image", page: 1 })).toBe("image");
  });
});
