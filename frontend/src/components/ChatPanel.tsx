import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown, { type Options } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMode, ChatTurn } from "../api/types";

export type ChatMessage = {
  id: number;
  author: "user" | "assistant";
  text: string;
};

const MODES: ChatMode[] = ["evidence", "guidance", "creative"];

// What the API takes as history: at most 50 turns of at most 16,000 characters
// each (ChatRequest and ChatTurn in the backend's api.py), of which the server
// reads only the most recent. So the page sends the tail of the conversation,
// an even number so it starts on a question, each turn cut to the limit: a
// long conversation or a long answer must not make the next question fail.
const HISTORY_TURNS_SENT = 24;
const HISTORY_TURN_MAX_CHARS = 16_000;

export function historyTurns(messages: ChatMessage[]): ChatTurn[] {
  return messages.slice(-HISTORY_TURNS_SENT).map((message) => ({
    role: message.author,
    content: message.text.slice(0, HISTORY_TURN_MAX_CHARS)
  }));
}

// A chat answer is model output, and the model has read text the app did not
// write (claims, quotes, source titles). So nothing in an answer may make the
// browser load or open an address: an image would send whatever its address
// carries to an outside host the moment it is shown. An image is shown as
// words, a link as its own text, and only the elements markdown itself
// produces are rendered; raw HTML already shows as text.
export const ANSWER_MARKDOWN: Options = {
  remarkPlugins: [remarkGfm],
  components: {
    img: ({ alt }) => (
      <span className="msg__removed">{alt ? `[image removed: ${alt}]` : "[image removed]"}</span>
    ),
    a: ({ children }) => <>{children}</>
  },
  // "a" and "img" are listed so the two renderers above run for them.
  allowedElements: [
    "p", "br", "strong", "em", "del", "code", "pre", "blockquote", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "table", "thead", "tbody", "tr", "th", "td",
    "sup", "section", "input", "a", "img"
  ],
  unwrapDisallowed: true
};

type ChatPanelProps = {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isSending?: boolean;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  suggestions?: string[];
  error?: string | null;
};

// The run-scoped chat, rendered as the center panel. Conversation history is
// held by the parent and sent with each request; the mode swaps a system
// instruction server-side.
const ChatPanel = ({
  messages,
  onSendMessage,
  isSending = false,
  mode,
  onModeChange,
  suggestions,
  error
}: ChatPanelProps) => {
  const [draft, setDraft] = useState("");
  const historyRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const formatMode = (value: ChatMode) => value.charAt(0).toUpperCase() + value.slice(1);

  useEffect(() => {
    const node = historyRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages, isSending]);

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (!modeMenuOpen) return;
      if (
        menuRef.current &&
        event.target instanceof Node &&
        !menuRef.current.contains(event.target)
      ) {
        setModeMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [modeMenuOpen]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [draft]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || isSending) return;
    setDraft("");
    onSendMessage(trimmed);
  };

  return (
    <main className="panel panel--main">
      <div className="panel__head">
        <h2>Chat</h2>
      </div>
      <div className="panel__body chat-scroll" ref={historyRef}>
        {messages.length === 0 && !isSending ? (
          <p className="chat-empty">
            Ask anything about this verification — answers come only from its claims, verdicts,
            and sources, never from outside knowledge.
          </p>
        ) : null}
        {messages.map((message) => (
          <div
            key={message.id}
            className={`msg ${message.author === "user" ? "msg--user" : "msg--assistant"}`}
          >
            {message.author === "user" ? (
              message.text
            ) : (
              <ReactMarkdown {...ANSWER_MARKDOWN}>{message.text}</ReactMarkdown>
            )}
          </div>
        ))}
        {isSending ? (
          <div className="typing" aria-label="Answer in progress">
            <span />
            <span />
            <span />
          </div>
        ) : null}
      </div>
      <div className="chat-bottom">
        {error ? <p className="error-text">{error}</p> : null}
        {messages.length <= 1 && !isSending && suggestions?.length ? (
          <div className="suggestions">
            {suggestions.map((suggestion) => (
              // Chips send immediately — pre-filling the input read as a dead
              // click (nothing visibly happened).
              <button
                key={suggestion}
                type="button"
                className="suggestion"
                onClick={() => onSendMessage(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}
        <form className="chat-input" onSubmit={handleSubmit}>
          <textarea
            ref={textareaRef}
            placeholder="Ask about this verification…"
            value={draft}
            rows={1}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            disabled={isSending}
          />
          <div className="chat-mode" ref={menuRef}>
            <button
              type="button"
              className="chat-mode__button"
              onClick={() => setModeMenuOpen((open) => !open)}
              aria-label="Select chat mode"
            >
              {formatMode(mode)} ▾
            </button>
            {modeMenuOpen ? (
              <div className="chat-mode__menu">
                {MODES.map((option) => (
                  <button
                    key={option}
                    type="button"
                    className={`chat-mode__option${option === mode ? " active" : ""}`}
                    onClick={() => {
                      onModeChange(option);
                      setModeMenuOpen(false);
                    }}
                  >
                    {formatMode(option)}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
          <button className="chat-input__send" type="submit" aria-label="Send" disabled={isSending}>
            ➤
          </button>
        </form>
        <p className="chat-disclaimer">Chat can make mistakes — check the quoted evidence.</p>
      </div>
    </main>
  );
};

export default ChatPanel;
