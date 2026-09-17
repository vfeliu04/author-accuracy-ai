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
    // A batch stopped in any other state is the same wait.
    expect(
      humanizeError(
        "TimeoutError: Batch msgbatch_01 still 'validating' after 86400s — it is still queued server-side; its id is retained, so a retry resumes it instead of paying twice"
      )
    ).toMatch(/reattaches to it/);
    expect(humanizeError("Your credit balance is too low to access the API")).toMatch(
      /out of credit/
    );
  });

  it("explains a link to a private network address and names the link", () => {
    const hint = humanizeError(
      "BlockedAddressError: Refusing to fetch 'https://10.0.0.5/admin': '10.0.0.5' resolves to a private or reserved network address"
    );
    expect(hint).toBe(
      "https://10.0.0.5/admin — That link points to a private network address, so it can't be opened."
    );
  });

  it("explains a link that isn't a web page or PDF", () => {
    const hint = humanizeError(
      "FetchError: Fetching 'https://example.org/archive.zip' failed: unsupported content type 'application/zip' (a source must be an HTML page or a PDF)"
    );
    expect(hint).toBe("https://example.org/archive.zip — That link isn't a web page or PDF.");
  });

  it("reports the status a site answered with", () => {
    const hint = humanizeError(
      "FetchError: Fetching 'https://example.org/gone' failed: the server answered HTTP 404"
    );
    expect(hint).toBe("https://example.org/gone — The site returned an error (404).");
  });

  it("explains a slow site for every timeout phrasing", () => {
    for (const error of [
      "FetchError: Fetching 'https://slow.example.org/page' timed out (30-second time budget)",
      "FetchError: Fetching 'https://slow.example.org/page' timed out after its 30-second time budget",
      "Could not read https://slow.example.org/page: exceeded the 60s deadline"
    ]) {
      expect(humanizeError(error)).toBe(
        "https://slow.example.org/page — The site took too long to respond."
      );
    }
  });

  it("explains a page with no readable text", () => {
    expect(
      humanizeError(
        "ThinPageError: https://app.example.org/ has no readable article text (5 characters extracted, at least 250 needed) — JavaScript-only pages are not supported"
      )
    ).toBe(
      "https://app.example.org/ — That page has no readable text. Pages that need JavaScript to show their content can't be read."
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
    expect(
      humanizeError(
        "FetchError: Fetching 'https://example.org/billing/faq' failed: the server answered HTTP 404"
      )
    ).toBe("https://example.org/billing/faq — The site returned an error (404).");
    expect(
      humanizeError(
        "BlockedAddressError: Refusing to fetch 'http://192.168.1.10/billing': '192.168.1.10' resolves to a private or reserved network address"
      )
    ).toBe(
      "http://192.168.1.10/billing — That link points to a private network address, so it can't be opened."
    );
    expect(
      humanizeError(
        "FetchError: Fetching 'https://docs.example.org/batch-processing' timed out (30-second time budget)"
      )
    ).toBe("https://docs.example.org/batch-processing — The site took too long to respond.");
    // A certificate failure is not a timeout, whatever the page is called.
    expect(
      humanizeError(
        "FetchError: Fetching 'https://www.irs.gov/filing/individuals/when-to-file/deadlines' failed: ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1010)"
      )
    ).toBe(
      "https://www.irs.gov/filing/individuals/when-to-file/deadlines — A secure connection to the site couldn't be established."
    );
  });

  // DNS and private-address failures repeat the link's host on its own, outside
  // the link; the words in that host must not pick the message either.
  it("does not let a word in a host the message repeats pick the message", () => {
    const notFound =
      "That site couldn't be found. Check the link for typos, or the internet connection.";
    expect(
      humanizeError(
        "FetchError: Fetching 'https://billing.example.org/report' failed: 'billing.example.org' could not be resolved ([Errno 8] nodename nor servname provided, or not known)"
      )
    ).toBe(`https://billing.example.org/report — ${notFound}`);
    expect(
      humanizeError(
        "FetchError: Fetching 'https://deadline.com/2024/05/box-office-report/' failed: 'deadline.com' could not be resolved ([Errno 8] nodename nor servname provided, or not known)"
      )
    ).toBe(`https://deadline.com/2024/05/box-office-report/ — ${notFound}`);
    expect(
      humanizeError(
        "FetchError: Fetching 'https://ebilling.example.net/faq' failed: 'ebilling.example.net' resolved to no addresses"
      )
    ).toBe(`https://ebilling.example.net/faq — ${notFound}`);
    expect(
      humanizeError(
        "BlockedAddressError: Refusing to fetch 'https://billing.corp.example/': 'billing.corp.example' resolves to a private or reserved network address"
      )
    ).toBe(
      "https://billing.corp.example/ — That link points to a private network address, so it can't be opened."
    );
    expect(
      humanizeError(
        "BlockedAddressError: Refusing to fetch 'https://deadline.example.org/' (redirected from 'https://example.org/go'): 'deadline.example.org' resolves to a private or reserved network address"
      )
    ).toBe(
      "https://deadline.example.org/ — That link points to a private network address, so it can't be opened."
    );
    // The link keeps its port; the host the server repeats does not.
    expect(
      humanizeError(
        "FetchError: Fetching 'https://billing.example.org:8443/x' failed: 'billing.example.org' could not be resolved ([Errno 8] nodename nor servname provided, or not known)"
      )
    ).toBe(`https://billing.example.org:8443/x — ${notFound}`);
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

  it("keeps a DNS failure on a host with a trigger word from sounding like a slow site", () => {
    // Offline, every link fails DNS; deadline.com must not read as a slow site.
    expect(
      humanizeError(
        "FetchError: Fetching 'https://deadline.com/' failed: 'deadline.com' could not be resolved ([Errno 8] nodename nor servname provided, or not known)"
      )
    ).not.toMatch(/too long|out of credit/);
  });

  it("marks a link the message cut short", () => {
    const shown = `https://example.org/${"a".repeat(180)}`;
    expect(
      humanizeError(`FetchError: Fetching '${shown}...' failed: the server answered HTTP 404`)
    ).toBe(`${shown}… — The site returned an error (404).`);
  });
});

// Each message below is exactly what the server stores for the failure.
describe("humanizeError explains every way a link can fail to open", () => {
  const cases: [string, string, string][] = [
    [
      "a host that doesn't resolve",
      "FetchError: Fetching 'https://nonexistent-host.example/x' failed: 'nonexistent-host.example' could not be resolved ([Errno 8] nodename nor servname provided, or not known)",
      "https://nonexistent-host.example/x — That site couldn't be found. Check the link for typos, or the internet connection."
    ],
    [
      "a host with no addresses",
      "FetchError: Fetching 'https://example.org/a' failed: 'example.org' resolved to no addresses",
      "https://example.org/a — That site couldn't be found. Check the link for typos, or the internet connection."
    ],
    [
      "a host whose DNS answer can't be read",
      "FetchError: Fetching 'https://weird.example.org/a' failed: DNS for 'weird.example.org' returned an unparseable address",
      "https://weird.example.org/a — That site couldn't be found. Check the link for typos, or the internet connection."
    ],
    [
      "a refused connection",
      "FetchError: Fetching 'https://example.org/down' failed: ConnectError: [Errno 61] Connection refused",
      "https://example.org/down — The site couldn't be reached, or it dropped the connection."
    ],
    [
      "a dropped connection",
      "FetchError: Fetching 'https://example.org/down' failed: ReadError: [Errno 54] Connection reset by peer",
      "https://example.org/down — The site couldn't be reached, or it dropped the connection."
    ],
    [
      "a server that hung up",
      "FetchError: Fetching 'https://example.org/down' failed: RemoteProtocolError: Server disconnected without sending a response.",
      "https://example.org/down — The site couldn't be reached, or it dropped the connection."
    ],
    [
      "a certificate that can't be verified",
      "FetchError: Fetching 'https://example.org/down' failed: ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1010)",
      "https://example.org/down — A secure connection to the site couldn't be established."
    ],
    [
      "a failed secure handshake",
      "FetchError: Fetching 'https://example.org/old-tls' failed: ConnectError: [SSL: TLSV1_ALERT_PROTOCOL_VERSION] tlsv1 alert protocol version (_ssl.c:1010)",
      "https://example.org/old-tls — A secure connection to the site couldn't be established."
    ],
    [
      "a page over the size cap",
      "FetchError: Fetching 'https://example.org/huge' failed: the response exceeds the 10,000,000-byte limit",
      "https://example.org/huge — That page is too large to read."
    ],
    [
      "a redirect loop",
      "FetchError: Fetching 'https://example.org/loop' failed: more than 5 redirects",
      "https://example.org/loop — That link redirects too many times to follow."
    ],
    [
      "a redirect to an address that isn't allowed",
      "FetchError: Fetching 'https://example.org/r' failed: refused redirect. Source URL 'ftp://example.org/x' must start with http:// or https://",
      "https://example.org/r — That link redirects to an address that can't be opened."
    ],
    [
      "a redirect to a malformed address",
      "FetchError: Fetching 'https://example.org/r' failed: the server sent an invalid redirect Location (For absolute URLs, path must be empty or begin with '/')",
      "https://example.org/r — That link redirects to an address that can't be opened."
    ],
    [
      "a redirect that names no address",
      "FetchError: Fetching 'https://example.org/r' failed: HTTP 301 redirect without a Location header",
      "https://example.org/r — That link redirects to an address that can't be opened."
    ],
    [
      "a PDF link that sent something else",
      "FetchError: Fetching 'https://example.org/file.pdf' failed: served as 'application/pdf' but the body is not a PDF",
      "https://example.org/file.pdf — The site says that link is a PDF, but the file it sent isn't one."
    ],
    [
      "a compression the app can't unpack",
      "FetchError: Fetching 'https://example.org/br' failed: unsupported Content-Encoding 'br'",
      "https://example.org/br — The site sent that page in a form that can't be read."
    ],
    [
      "stacked compressions",
      "FetchError: Fetching 'https://example.org/gz' failed: unsupported stacked Content-Encoding 'gzip, gzip, gzip, gzip, ...'",
      "https://example.org/gz — The site sent that page in a form that can't be read."
    ],
    [
      "a corrupt compressed body",
      "FetchError: Fetching 'https://example.org/gz' failed: DecodingError: Error -3 while decompressing data: incorrect header check",
      "https://example.org/gz — The site sent that page in a form that can't be read."
    ],
    [
      "a page that isn't valid UTF-8 and names no charset",
      "ValueError: https://example.org/latin is not valid UTF-8 and has no byte-order mark or <meta> charset declaration in its first 4096 bytes — decode it with the HTTP Content-Type charset and pass text instead",
      "https://example.org/latin — That page's text is in an encoding that can't be read."
    ],
    [
      "a page that declares an unknown charset",
      "ValueError: https://example.org/latin declares an unknown charset 'x-bogus-8'",
      "https://example.org/latin — That page's text is in an encoding that can't be read."
    ],
    [
      "a page too large or complex to read in time",
      "ExtractionTimeoutError: https://example.org/huge took longer than 60 seconds to read (the page is too large or complex)",
      "https://example.org/huge — That page is too large or complex to read in time."
    ],
    [
      "a page too large or complex to read in time, reached by a redirect",
      "ExtractionTimeoutError: 'https://www.example.org/huge/' (redirected from 'https://example.org/huge'): https://www.example.org/huge/ took longer than 60 seconds to read (the page is too large or complex)",
      "https://www.example.org/huge/ — That page is too large or complex to read in time."
    ],
    [
      "a page whose reader stopped without a result",
      "RuntimeError: https://www.who.int/facts could not be read: the reader process exited without a result",
      "https://www.who.int/facts — Reading that page failed unexpectedly. Retrying may help; if it keeps failing, remove that link."
    ],
    [
      "a page whose reader failed with an unexpected error",
      "RuntimeError: https://www.who.int/facts could not be read: RecursionError: maximum recursion depth exceeded while calling a Python object",
      "https://www.who.int/facts — Reading that page failed unexpectedly. Retrying may help; if it keeps failing, remove that link."
    ],
    [
      "a page whose reader failed with an error that mentions a timeout",
      "RuntimeError: https://www.who.int/facts could not be read: OSError: [Errno 60] Operation timed out",
      "https://www.who.int/facts — Reading that page failed unexpectedly. Retrying may help; if it keeps failing, remove that link."
    ],
    [
      "a page whose reader stopped without a result, reached by a redirect",
      "RuntimeError: 'https://who.int/facts/' (redirected from 'https://www.who.int/facts'): https://who.int/facts/ could not be read: the reader process exited without a result",
      "https://who.int/facts/ — Reading that page failed unexpectedly. Retrying may help; if it keeps failing, remove that link."
    ],
    [
      "a page whose reader failed with an unexpected error, reached by a redirect",
      "RuntimeError: 'https://who.int/facts/' (redirected from 'https://www.who.int/facts'): https://who.int/facts/ could not be read: RecursionError: maximum recursion depth exceeded while calling a Python object",
      "https://who.int/facts/ — Reading that page failed unexpectedly. Retrying may help; if it keeps failing, remove that link."
    ],
    [
      "a thin page reached by a redirect",
      "ThinPageError: 'https://www.example.org/a/' (redirected from 'https://example.org/a'): https://www.example.org/a/ has no readable article text (0 characters extracted, at least 250 needed) — JavaScript-only pages are not supported",
      "https://www.example.org/a/ — That page has no readable text. Pages that need JavaScript to show their content can't be read."
    ],
    [
      "an undecodable page reached by a redirect",
      "ValueError: 'https://www.example.org/latin' (redirected from 'https://example.org/latin'): https://www.example.org/latin declares an unknown charset 'x-bogus-8'",
      "https://www.example.org/latin — That page's text is in an encoding that can't be read."
    ]
  ];

  for (const [name, error, expected] of cases) {
    it(`explains ${name}`, () => {
      expect(humanizeError(error)).toBe(expected);
    });
  }

  it("keeps provider and registry failures on their own explanations", () => {
    expect(
      humanizeError(
        "RuntimeError: Crossref gave no answer after 3 attempts (https://api.crossref.org/works?query=x): ConnectError: [Errno 61] Connection refused"
      )
    ).toMatch(/publication registry/);
    expect(
      humanizeError(
        "RuntimeError: Crossref gave no answer after 3 attempts (https://api.crossref.org/works?query=x): ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
      )
    ).toMatch(/publication registry/);
    expect(humanizeError("APIConnectionError: Connection error.")).toMatch(/laptop sleep/);
    expect(humanizeError("Your credit balance is too low to access the API")).toMatch(
      /out of credit/
    );
  });

  it("describes a network failure as a site's only when the message names a link", () => {
    expect(
      humanizeError("ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    ).toBeNull();
    expect(humanizeError("OSError: 'localhost' could not be resolved")).toBeNull();
  });
});

// The server quotes a site's Content-Type or Content-Encoding value in full, and
// a site may send about 100 KiB of headers. "https://" then a long run of
// closing brackets is a link whose every bracket the trimming looks at; the page
// showing the failure reads the message several times while it renders.
describe("reading a failure that quotes a huge header", () => {
  const link = "http://enc.example:8080/page";
  // Shorter than a real header could be: trimming that re-reads the link for
  // every bracket takes seconds on 12,000 of them, well past the budget, but
  // minutes on 100 KiB, and a test cannot interrupt synchronous code, so a
  // run that long would hang instead of failing.
  const brackets = 12_000;
  const header = 100 * 1024;
  const cases: [string, string, string][] = [
    [
      "a Content-Encoding",
      `FetchError: Fetching '${link}' failed: unsupported Content-Encoding 'x-https://a${")".repeat(brackets)}'`,
      `${link} — The site sent that page in a form that can't be read.`
    ],
    [
      "a content type",
      `FetchError: Fetching '${link}' failed: unsupported content type 'x-https://a${"]".repeat(brackets)}' (a source must be an HTML page or a PDF)`,
      `${link} — That link isn't a web page or PDF.`
    ],
    [
      "a header repeating a word a hint looks for",
      `FetchError: Fetching '${link}' failed: unsupported Content-Encoding '${"TimeoutError ".repeat(header / 13)}'`,
      `${link} — The site sent that page in a form that can't be read.`
    ]
  ];

  for (const [name, error, expected] of cases) {
    it(`takes well under a second for ${name}`, () => {
      const started = performance.now();
      expect(humanizeError(error)).toBe(expected);
      expect(linksNamedIn(error, [link, "https://example.org/other"])).toEqual([link]);
      expect(performance.now() - started).toBeLessThan(500);
    });
  }
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

  // The server quotes a link the way Python prints it, doubling each backslash
  // a query string may keep ("?q=C:\Users"); the messages below are exactly its.
  it("finds a link whose backslashes the message doubled, whole or cut short", () => {
    const added = String.raw`https://www.example.org/search?q=C:\Users\data`;
    const shorter = "https://www.example.org/search";
    expect(
      linksNamedIn(
        String.raw`FetchError: Fetching 'https://www.example.org/search?q=C:\\Users\\data' failed: the server answered HTTP 404`,
        [shorter, added]
      )
    ).toEqual([added]);
    const long = String.raw`https://www.example.org/s?q=C:\Users\a` + "a".repeat(299);
    expect(
      linksNamedIn(
        String.raw`FetchError: Fetching 'https://www.example.org/s?q=C:\\Users\\` +
          `${"a".repeat(163)}...' failed: the server answered HTTP 404`,
        [other, long]
      )
    ).toEqual([long]);
  });

  it("names nothing when no added link appears", () => {
    expect(linksNamedIn("APIConnectionError: Connection error.", [wiki, other])).toEqual([]);
    expect(linksNamedIn(`Could not read ${wiki}: HTTP 404`, [])).toEqual([]);
  });
});
