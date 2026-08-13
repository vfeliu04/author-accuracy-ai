import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

// Confirms the jsdom environment and jest-dom matchers are wired up, so the
// vitest CI step is exercising a real render rather than an empty suite.
describe("test harness", () => {
  it("renders into jsdom with jest-dom matchers", () => {
    render(<h1>Author Accuracy</h1>);
    expect(screen.getByRole("heading", { name: "Author Accuracy" })).toBeInTheDocument();
  });
});
