import { describe, expect, it } from "vitest";
import { locateQuote } from "./quote";

// The slice of the ORIGINAL text a located quote covers.
function covered(text: string, quote: string): string | null {
  const range = locateQuote(text, quote);
  return range ? text.slice(range.start, range.end) : null;
}

describe("locateQuote", () => {
  it("returns the original offsets of an exact quote", () => {
    const text = "Intro. Hunger fell by 12 percent in 2023. Outro.";
    expect(locateQuote(text, "Hunger fell by 12 percent")).toEqual({ start: 7, end: 32 });
  });

  it("ignores case", () => {
    expect(covered("HUNGER Fell sharply", "hunger fell")).toBe("HUNGER Fell");
  });

  it("matches curly quotes and apostrophes against straight ones, in both directions", () => {
    const curly = "The minister said “we’ll act” today.";
    expect(covered(curly, `said "we'll act"`)).toBe("said “we’ll act”");
    expect(covered(`it's "fine" now`, "it’s “fine”")).toBe(`it's "fine"`);
  });

  it("treats en and em dashes as hyphens", () => {
    expect(covered("rose 10–12% — a record", "10-12% - a record")).toBe(
      "10–12% — a record"
    );
  });

  it("collapses non-breaking spaces, tabs and whitespace runs", () => {
    const text = "a total of  733\tmillion people";
    expect(covered(text, "a total of 733 million")).toBe("a total of  733\tmillion");
  });

  it("matches a quote across a line break", () => {
    const text = "about 733\nmillion people went hungry";
    expect(covered(text, "733 million people")).toBe("733\nmillion people");
  });

  it("spans a paragraph break and maps both ends back to the original text", () => {
    const text = "First paragraph ends here.\n\nSecond paragraph begins now.";
    const range = locateQuote(text, "ends here. Second paragraph");
    expect(range).toEqual({
      start: text.indexOf("ends here"),
      end: text.indexOf("Second paragraph") + "Second paragraph".length
    });
  });

  it("ignores whitespace around the quote itself", () => {
    expect(covered("x hunger fell y", "  hunger fell\n")).toBe("hunger fell");
  });

  it("keeps offsets exact after characters whose lowercase form is longer", () => {
    // "İ" lowercases to two code units; offsets must still index the original.
    expect(covered("İstanbul grew fast", "grew fast")).toBe("grew fast");
    expect(covered("İstanbul grew fast", "İstanbul")).toBe("İstanbul");
  });

  it("keeps offsets exact after astral characters", () => {
    expect(covered("\u{1F4A7} water use rose", "water use")).toBe("water use");
  });

  it("returns the first occurrence", () => {
    expect(locateQuote("water, water", "water")).toEqual({ start: 0, end: 5 });
  });

  it("returns null when the quote is absent or blank", () => {
    expect(locateQuote("hunger fell", "hunger rose")).toBeNull();
    expect(locateQuote("hunger fell", "   ")).toBeNull();
    expect(locateQuote("", "hunger")).toBeNull();
  });
});
