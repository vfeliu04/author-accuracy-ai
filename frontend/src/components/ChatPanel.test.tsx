import { render } from "@testing-library/react";
import ReactMarkdown from "react-markdown";
import { describe, expect, it } from "vitest";
import ChatPanel, { ANSWER_MARKDOWN, historyTurns, type ChatMessage } from "./ChatPanel";

// Every attribute through which an element can load, open or post to an address.
const ADDRESS_ATTRIBUTES =
  "[src],[href],[srcset],[poster],[cite],[action],[formaction],[data],[ping]";
const ADDRESS_ELEMENTS =
  "img,a,video,audio,source,picture,iframe,object,embed,link,form,area,base";

function renderAnswer(text: string) {
  const messages: ChatMessage[] = [
    { id: 1, author: "user", text: "What does the analysis say?" },
    { id: 2, author: "assistant", text }
  ];
  const view = render(
    <ChatPanel
      messages={messages}
      onSendMessage={() => {}}
      mode="evidence"
      onModeChange={() => {}}
    />
  );
  const answer = view.container.querySelector(".msg--assistant") as HTMLElement;
  return { ...view, answer };
}

// An answer is model output, and the model has read text the app did not
// write. The first thirteen made the markdown renderer emit an element that
// loads or opens an address until ChatPanel limited it. The three raw-HTML
// ones never did (react-markdown shows raw HTML as text): they are here so
// that a later plugin or option cannot quietly change that.
const HOSTILE: Array<[string, string]> = [
  ["an image", "Done.\n\n![usage badge](https://badges.outside.example/b.png?d=the-question)"],
  ["a link", "See [the full dataset](https://outside.example/data)."],
  ["a reference-style image", "![badge][b]\n\n[b]: https://outside.example/b.png"],
  ["a collapsed reference image", "![badge][]\n\n[badge]: https://outside.example/b.png"],
  ["an image inside a link", "[![badge](https://outside.example/b.png)](https://outside.example/go)"],
  [
    "an image in a table cell",
    "| a | b |\n| --- | --- |\n| ![x](https://outside.example/b.png) | 1 |"
  ],
  ["an image in a footnote", "Text.[^1]\n\n[^1]: ![x](https://outside.example/b.png)"],
  ["a bare address", "Visit https://outside.example/page for more."],
  ["a www address", "Visit www.outside.example/page for more."],
  ["an email address", "Write to desk@outside.example today."],
  ["an address in angle brackets", "<https://outside.example/ü>"],
  ["a protocol-relative link", "[here](//outside.example/x)"],
  ["a javascript link", "[here](javascript:alert(1))"],
  ["raw image HTML", '<img src="https://outside.example/b.png" alt="badge">'],
  ["raw link HTML", '<a href="https://outside.example/">here</a>'],
  [
    "raw video HTML",
    '<video poster="https://outside.example/p.png"><source srcset="https://outside.example/v"></video>'
  ]
];

describe("historyTurns", () => {
  const conversation = (exchanges: number): ChatMessage[] =>
    Array.from({ length: exchanges * 2 }, (_, index) => ({
      id: index,
      author: index % 2 === 0 ? "user" : "assistant",
      text: `message ${index}`
    }));

  it("sends a short conversation whole, in order", () => {
    expect(historyTurns(conversation(2))).toEqual([
      { role: "user", content: "message 0" },
      { role: "assistant", content: "message 1" },
      { role: "user", content: "message 2" },
      { role: "assistant", content: "message 3" }
    ]);
  });

  it("sends only the most recent turns of a long one, starting on a question", () => {
    // The API refuses more than 50 history turns: 30 exchanges are 60.
    const turns = historyTurns(conversation(30));
    expect(turns).toHaveLength(24);
    expect(turns[0]).toEqual({ role: "user", content: "message 36" });
    expect(turns[23]).toEqual({ role: "assistant", content: "message 59" });
  });

  it("cuts a turn longer than the API accepts", () => {
    const long: ChatMessage[] = [{ id: 1, author: "assistant", text: "x".repeat(20_000) }];
    expect(historyTurns(long)[0].content).toHaveLength(16_000);
  });

  it("never cuts an emoji in half", () => {
    // The emoji's two UTF-16 units sit on either side of the limit.
    const text = "x".repeat(15_999) + "😀 and more";
    const [turn] = historyTurns([{ id: 1, author: "user", text }]);
    expect(turn.content).toBe("x".repeat(15_999));
    const whole = "x".repeat(15_998) + "😀 and more";
    expect(historyTurns([{ id: 1, author: "user", text: whole }])[0].content).toBe(
      "x".repeat(15_998) + "😀"
    );
  });
});

describe("ChatPanel answers", () => {
  it.each(HOSTILE)("render %s without any element that carries an address", (_name, text) => {
    const { answer } = renderAnswer(text);
    expect(answer.querySelector(ADDRESS_ATTRIBUTES)).toBeNull();
    expect(answer.querySelector(ADDRESS_ELEMENTS)).toBeNull();
  });

  it("say that an image was removed, and keep its description", () => {
    const { answer } = renderAnswer(
      "Five claims rely on it.\n\n![usage badge](https://badges.outside.example/b.png)\n\n![](https://outside.example/c.png)"
    );
    expect(answer).toHaveTextContent("Five claims rely on it.");
    expect(answer).toHaveTextContent("[image removed: usage badge]");
    expect(answer).toHaveTextContent("[image removed]");
    expect(answer).not.toHaveTextContent("outside.example");
  });

  it("keep a link's words and drop its target", () => {
    const { answer } = renderAnswer("See [the full dataset](https://outside.example/data) first.");
    expect(answer).toHaveTextContent("See the full dataset first.");
    expect(answer).not.toHaveTextContent("outside.example");
  });

  it("leave no stray arrow where a footnote linked back to the text", () => {
    const { answer } = renderAnswer("Hunger fell.[^1]\n\n[^1]: By the index's own count.");
    expect(answer).toHaveTextContent("By the index's own count.");
    expect(answer).not.toHaveTextContent("↩");
  });

  it("keep an address the answer wrote out, as plain text", () => {
    const { answer } = renderAnswer("Visit https://outside.example/page or www.outside.example.");
    expect(answer).toHaveTextContent("Visit https://outside.example/page or www.outside.example.");
  });

  it("still render ordinary markdown", () => {
    const { answer } = renderAnswer(
      [
        "## Verdicts",
        "",
        "**Two** claims were *contradicted*; `p.3` has the quote.",
        "",
        "- first",
        "- second",
        "",
        "> the source says so",
        "",
        "| claim | verdict |",
        "| --- | --- |",
        "| one | SUPPORTED |"
      ].join("\n")
    );
    for (const tag of ["h2", "strong", "em", "code", "ul", "li", "blockquote", "table", "td"]) {
      expect(answer.querySelector(tag), tag).not.toBeNull();
    }
    expect(answer).toHaveTextContent("SUPPORTED");
  });

  it("show the user's own message as plain text", () => {
    const messages: ChatMessage[] = [
      { id: 1, author: "user", text: "![x](https://outside.example/b.png) **not bold**" }
    ];
    const { container } = render(
      <ChatPanel
        messages={messages}
        onSendMessage={() => {}}
        mode="evidence"
        onModeChange={() => {}}
      />
    );
    const message = container.querySelector(".msg--user") as HTMLElement;
    expect(message).toHaveTextContent("![x](https://outside.example/b.png) **not bold**");
    expect(message.querySelector("img, a, strong")).toBeNull();
  });

  it("drop any element markdown itself does not produce, whatever a plugin adds", () => {
    // No plugin in use emits one today. This one stands in for a later
    // addition: it appends a <video poster> with a <source srcset> inside.
    const addVideo = () => (tree: { children: unknown[] }) => {
      tree.children.push({
        type: "element",
        tagName: "video",
        properties: { poster: "https://outside.example/p.png" },
        children: [
          {
            type: "element",
            tagName: "source",
            properties: { srcSet: "https://outside.example/v" },
            children: []
          },
          { type: "text", value: "words inside it" }
        ]
      });
    };
    const { container } = render(
      <ReactMarkdown {...ANSWER_MARKDOWN} rehypePlugins={[addVideo]}>
        An answer.
      </ReactMarkdown>
    );
    expect(container.querySelector(ADDRESS_ATTRIBUTES)).toBeNull();
    expect(container.querySelector(ADDRESS_ELEMENTS)).toBeNull();
    expect(container).toHaveTextContent("An answer.");
    expect(container).toHaveTextContent("words inside it");
  });
});
