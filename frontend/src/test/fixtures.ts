import type { PageSnapshot, ScannedReference, VideoProvenance } from "../api/types";

// A cited work as the scan returns it: every field the dialog reads, nothing
// printed and nothing found, with `over` setting what a test is about.
export function scannedReference(over: Partial<ScannedReference> = {}): ScannedReference {
  return {
    title: null,
    authors: [],
    year: null,
    doi: null,
    url: null,
    label: null,
    retrievability: "unknown",
    suggested_url: null,
    ...over
  };
}

// What a YouTube video declares about itself, with `over` setting what a
// test is about.
export function videoProvenance(over: Partial<VideoProvenance> = {}): VideoProvenance {
  return {
    id: "dQw4w9WgXcQ",
    channel_id: "UCabc",
    channel_verified: true,
    duration_seconds: 300,
    embeddable: true,
    captions: { kind: "automatic", language: "en" },
    ...over
  };
}

// A YouTube video's page snapshot: two captioned windows a player can be
// pointed at. `document` and `provenance` are full objects in PageSnapshot,
// so a test that changes either passes the whole replacement in `over`.
export function videoSnapshot(over: Partial<PageSnapshot> = {}): PageSnapshot {
  return {
    schema: 1,
    document: {
      title: "How Water Crises Start",
      sections: [
        {
          title: "0:00–1:15",
          page: null,
          text: "An introduction.",
          start_seconds: 0,
          end_seconds: 75
        },
        {
          title: "1:15–2:29",
          page: null,
          text: "Two billion people lack safe water at home.",
          start_seconds: 75,
          end_seconds: 149
        }
      ]
    },
    provenance: {
      url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      final_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      fetched_at: "2026-09-01T10:00:00Z",
      title: "How Water Crises Start",
      authors: [],
      publisher: "Example Channel",
      publication_date: "2024-03-01",
      doi: null,
      scholarly: false,
      video: videoProvenance()
    },
    ...over
  };
}
