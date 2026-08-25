import { describe, expect, it } from "vitest";
import { emojiFor } from "./emoji";

describe("emojiFor", () => {
  it("is deterministic for the same seed", () => {
    expect(emojiFor("Water Stress Report")).toBe(emojiFor("Water Stress Report"));
  });

  it("always returns a non-empty icon, even for empty seeds", () => {
    expect(emojiFor("").length).toBeGreaterThan(0);
    expect(emojiFor("x").length).toBeGreaterThan(0);
  });
});
