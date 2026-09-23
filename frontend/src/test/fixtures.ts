import type { ScannedReference } from "../api/types";

// A cited work as the scan returns it: every field the dialog reads, nothing
// printed and nothing found, with `over` setting what a test is about.
export function scannedReference(over: Partial<ScannedReference> = {}): ScannedReference {
  return {
    title: null,
    authors: [],
    year: null,
    doi: null,
    url: null,
    entry: "An entry as printed.",
    retrievability: "unknown",
    suggested_url: null,
    ...over
  };
}
