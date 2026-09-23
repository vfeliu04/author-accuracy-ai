import { describe, expect, it } from "vitest";
import type { ScannedReference } from "../api/types";
import {
  alreadyAdded,
  cleanDoi,
  doiInUrl,
  familyNames,
  normalizeTitle,
  stem
} from "./references";

function ref(over: Partial<ScannedReference> = {}): ScannedReference {
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

describe("stem", () => {
  it("drops the last extension only", () => {
    expect(stem("Coastal_Brief.pdf")).toBe("Coastal_Brief");
    expect(stem("report.v2.pdf")).toBe("report.v2");
    expect(stem("no-extension")).toBe("no-extension");
    expect(stem(".hidden")).toBe(".hidden");
  });
});

describe("cleanDoi", () => {
  it("strips the doi.org and doi: prefixes the entry may carry", () => {
    expect(cleanDoi("https://doi.org/10.1000/ABC")).toBe("10.1000/ABC");
    expect(cleanDoi("http://dx.doi.org/10.1000/abc")).toBe("10.1000/abc");
    expect(cleanDoi("doi: 10.1000/abc")).toBe("10.1000/abc");
    expect(cleanDoi("  10.1000/abc  ")).toBe("10.1000/abc");
  });

  it("rejects anything that is not DOI-shaped", () => {
    expect(cleanDoi("10.12/abc")).toBeNull();
    expect(cleanDoi("not a doi")).toBeNull();
    expect(cleanDoi("10.1000/")).toBeNull();
    expect(cleanDoi("")).toBeNull();
  });
});

describe("doiInUrl", () => {
  it("finds the DOI in doi.org, dx.doi.org and publisher links", () => {
    expect(doiInUrl("https://doi.org/10.1016/j.heliyon.2024.e34730")).toBe(
      "10.1016/j.heliyon.2024.e34730"
    );
    expect(doiInUrl("http://dx.doi.org/10.1000/xyz?ref=1#frag")).toBe("10.1000/xyz");
    expect(doiInUrl("https://link.springer.com/article/10.1007/s00382-020-05201-4")).toBe(
      "10.1007/s00382-020-05201-4"
    );
    expect(doiInUrl("https://onlinelibrary.wiley.com/doi/full/10.1002%2Fjoc.7154")).toBe(
      "10.1002/joc.7154"
    );
  });

  it("finds nothing in a link without one, and never throws", () => {
    expect(doiInUrl("https://example.org/report")).toBeNull();
    expect(doiInUrl("https://example.org/v10.1000/abc")).toBeNull(); // not a path segment
    expect(doiInUrl("https://example.org/?doi=10.1000/abc")).toBeNull(); // not in the path
    expect(doiInUrl("not a link")).toBeNull();
  });
});

describe("normalizeTitle", () => {
  it("lowercases and collapses runs of punctuation, spacing and underscores", () => {
    expect(normalizeTitle("Water Scarcity: A Review (2nd ed.)")).toBe(
      "water scarcity a review 2nd ed"
    );
    expect(normalizeTitle("  Global_Hunger--Report  ")).toBe("global hunger report");
  });

  it("keeps letters of every script, and reads a decomposed accent as the letter", () => {
    expect(normalizeTitle("L'Étude — des eaux")).toBe("l étude des eaux");
    expect(normalizeTitle("E\u0301tude")).toBe(normalizeTitle("\u00c9tude"));
    expect(normalizeTitle("気候変動の影響")).toBe("気候変動の影響");
  });
});

describe("familyNames", () => {
  it("reads the family name from the forms a reference list prints", () => {
    expect(
      familyNames([
        "Smith, J.",
        "J. Jones",
        "Ann Marie Brown",
        "van der Berg, A.",
        "Wang JP",
        "Kim et al.",
        "Lee J",
        "Li, X."
      ])
    ).toEqual(["Smith", "Jones", "Brown", "van der Berg", "Wang", "Kim", "Lee", "Li"]);
  });

  it("skips blank authors", () => {
    expect(familyNames(["", "  ", "et al."])).toEqual([]);
  });

  // The "et al." rule used to start with `[,\s]*` before `\b`, which
  // backtracks from every position of a long run of separators: 6 s on a
  // 60,000-character author, run again on every source or link change. The
  // server caps a name at 200 characters (references.py AUTHOR_MAX_CHARS) and
  // the client at 500; the pattern must be linear on its own all the same.
  it("takes under 50 ms on a 60,000-character author, whatever it holds", () => {
    const inputs = [",".repeat(60_000) + "x", "A" + " ".repeat(60_000) + "B", "Smith, J.".repeat(7_000)];
    for (const input of inputs) {
      const started = performance.now();
      familyNames([input]);
      expect(performance.now() - started).toBeLessThan(50);
    }
  });
});

describe("alreadyAdded", () => {
  describe("by DOI", () => {
    const cited = ref({ doi: "10.1000/abc" });

    it("matches a link carrying the DOI in doi.org, dx.doi.org or publisher form", () => {
      for (const link of [
        "https://doi.org/10.1000/ABC",
        "https://dx.doi.org/10.1000/abc",
        "https://link.springer.com/article/10.1000/abc"
      ]) {
        expect(alreadyAdded(cited, [], [link])).toBe(link);
      }
    });

    it("accepts a DOI the entry printed with its prefix", () => {
      expect(
        alreadyAdded(ref({ doi: "https://doi.org/10.1000/abc" }), [], ["https://doi.org/10.1000/abc"])
      ).toBe("https://doi.org/10.1000/abc");
    });

    it("does not match another DOI, a malformed one, or a DOI outside the path", () => {
      expect(alreadyAdded(cited, [], ["https://doi.org/10.1000/abd"])).toBeNull();
      expect(alreadyAdded(ref({ doi: "abc" }), [], ["https://example.org/abc"])).toBeNull();
      expect(alreadyAdded(cited, [], ["https://example.org/?doi=10.1000/abc"])).toBeNull();
    });
  });

  describe("by canonical URL", () => {
    it("matches the printed URL through the dialog's own link rules, #fragment included", () => {
      const cited = ref({ url: "https://example.org/paper#abstract" });
      expect(alreadyAdded(cited, [], ["https://example.org/paper"])).toBe(
        "https://example.org/paper"
      );
      expect(alreadyAdded(cited, [], ["https://EXAMPLE.org/paper#top"])).toBe(
        "https://EXAMPLE.org/paper#top"
      );
    });

    it("matches the suggested copy too, so an added suggestion leaves the list", () => {
      const cited = ref({ suggested_url: "https://europepmc.org/articles/PMC1?pdf=render" });
      expect(alreadyAdded(cited, [], ["https://europepmc.org/articles/PMC1?pdf=render"])).toBe(
        "https://europepmc.org/articles/PMC1?pdf=render"
      );
    });

    it("does not match a different query string, or an address that isn't a link", () => {
      expect(
        alreadyAdded(ref({ url: "https://example.org/paper?v=2" }), [], ["https://example.org/paper"])
      ).toBeNull();
      expect(
        alreadyAdded(ref({ url: "www.example.org/paper" }), [], ["https://www.example.org/paper"])
      ).toBeNull();
    });
  });

  describe("by title", () => {
    const cited = ref({ title: "Water scarcity in the Mediterranean basin" });

    it("matches a file named after the title, in either direction of containment", () => {
      expect(alreadyAdded(cited, ["Water_scarcity_in_the_Mediterranean_basin.pdf"], [])).toBe(
        "Water_scarcity_in_the_Mediterranean_basin.pdf"
      );
      // The file name holds the title and more.
      expect(alreadyAdded(cited, ["FAO-2024-water-scarcity-in-the-mediterranean-basin-full.pdf"], [])).toBe(
        "FAO-2024-water-scarcity-in-the-mediterranean-basin-full.pdf"
      );
      // The title holds the file name and more.
      const longer = ref({
        title: "Global assessment of water scarcity in the Mediterranean basin and beyond"
      });
      expect(alreadyAdded(longer, ["water scarcity in the Mediterranean basin.pdf"], [])).toBe(
        "water scarcity in the Mediterranean basin.pdf"
      );
    });

    it("never matches on fewer than four tokens, so generic short titles stay listed", () => {
      expect(alreadyAdded(ref({ title: "Annual Report" }), ["Annual_Report.pdf"], [])).toBeNull();
      expect(alreadyAdded(ref({ title: "The 2020 report" }), ["the-2020-report.pdf"], [])).toBeNull();
      // A one-word file name is contained in almost any title.
      expect(alreadyAdded(cited, ["basin.pdf"], [])).toBeNull();
      expect(alreadyAdded(ref({ title: "Introduction" }), ["Introduction.pdf"], [])).toBeNull();
    });

    it("matches whole words only", () => {
      expect(
        alreadyAdded(ref({ title: "Global hunger report 2024" }), ["global_hunger_reports_2024.pdf"], [])
      ).toBeNull();
    });
  });

  describe("by family name and year", () => {
    const cited = ref({ authors: ["Smith, J.", "Jones, K."], year: 2020 });

    it("matches a file stem naming an author and the year", () => {
      const named = ["Smith_et_al_2020.pdf", "Smith2020.pdf", "smith-2020-water.pdf", "Jones_2020.pdf"];
      for (const name of named) {
        expect(alreadyAdded(cited, [name], [])).toBe(name);
      }
    });

    it("matches a multi-word family name written together or apart", () => {
      const berg = ref({ authors: ["van der Berg, A."], year: 2019 });
      expect(alreadyAdded(berg, ["vanderBerg2019.pdf"], [])).toBe("vanderBerg2019.pdf");
      expect(alreadyAdded(berg, ["van_der_Berg_2019.pdf"], [])).toBe("van_der_Berg_2019.pdf");
    });

    it("needs both the name as a whole word and the year as a whole number", () => {
      expect(alreadyAdded(cited, ["Smith_2021.pdf"], [])).toBeNull();
      expect(alreadyAdded(cited, ["Smithson_2020.pdf"], [])).toBeNull();
      expect(alreadyAdded(cited, ["Smith_20201.pdf"], [])).toBeNull();
      expect(alreadyAdded(cited, ["2020_report.pdf"], [])).toBeNull();
      expect(alreadyAdded(ref({ authors: ["Smith, J."], year: null }), ["Smith_2020.pdf"], [])).toBeNull();
      expect(alreadyAdded(ref({ authors: [], year: 2020 }), ["Smith_2020.pdf"], [])).toBeNull();
    });
  });

  it("takes the rules in order and answers with the first match", () => {
    const cited = ref({
      doi: "10.1000/abc",
      title: "Water scarcity in the Mediterranean basin",
      authors: ["Smith, J."],
      year: 2020
    });
    expect(
      alreadyAdded(
        cited,
        ["Water_scarcity_in_the_Mediterranean_basin.pdf", "Smith_2020.pdf"],
        ["https://doi.org/10.1000/abc"]
      )
    ).toBe("https://doi.org/10.1000/abc");
    expect(
      alreadyAdded(cited, ["Smith_2020.pdf", "Water_scarcity_in_the_Mediterranean_basin.pdf"], [])
    ).toBe("Water_scarcity_in_the_Mediterranean_basin.pdf");
    expect(alreadyAdded(cited, [], [])).toBeNull();
  });
});
