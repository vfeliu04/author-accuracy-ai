import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMode } from "../api/types";

export type ChatMessage = {
  id: number;
  author: "user" | "assistant";
  text: string;
};

const MODES: ChatMode[] = ["evidence", "guidance", "creative"];

function AiAvatar() {
  return <div className="chat__avatar chat__avatar--ai" aria-hidden="true">A</div>;
}
function UserAvatar() {
  return <div className="chat__avatar chat__avatar--user" aria-hidden="true">U</div>;
}

type ChatPanelProps = {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isSending?: boolean;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  suggestions?: string[];
};

// The report-scoped chat. Conversation history is held by the parent and sent
// with each request; the mode just swaps a system instruction server-side.
const ChatPanel = ({
  messages,
  onSendMessage,
  isSending = false,
  mode,
  onModeChange,
  suggestions
}: ChatPanelProps) => {
  const [draft, setDraft] = useState("");
  const historyRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
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
      if (menuRef.current && event.target instanceof Node && !menuRef.current.contains(event.target)) {
        setModeMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [modeMenuOpen]);

  useEffect(() => {
    document.body.style.overflow = isFullscreen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [isFullscreen]);

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
    <article className={`card card--chat${isFullscreen ? " chat--fullscreen" : ""}`}>
      <header className="card__header card__header--chat">
        <h2>Chat</h2>
        <button
          type="button"
          className="chat__fullscreen-button"
          onClick={() => setIsFullscreen((prev) => !prev)}
          aria-label={isFullscreen ? "Exit full screen" : "Expand chat"}
        >
          {isFullscreen ? "✕" : "⛶"}
        </button>
      </header>
      <div className="chat__history" ref={historyRef}>
        {messages.map((message) => {
          const isUser = message.author === "user";
          return (
            <div
              key={message.id}
              className={`chat__message-row ${isUser ? "chat__message-row--user" : "chat__message-row--ai"}`}
            >
              {!isUser && <AiAvatar />}
              <div className={`chat__bubble ${isUser ? "chat__bubble--user" : "chat__bubble--ai"}`}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
              </div>
              {isUser && <UserAvatar />}
            </div>
          );
        })}
        {isSending ? (
          <div className="chat__typing-indicator">
            <span />
            <span />
            <span />
          </div>
        ) : null}
      </div>
      {messages.length <= 1 && !isSending && (
        <div className="chat__suggestions">
          {(suggestions ?? []).map((s) => (
            <button key={s} type="button" className="chat__suggestion-chip" onClick={() => setDraft(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
      <form className="chat__composer-inner" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="chat__input"
          placeholder="Ask about your report..."
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
        <div className="chat__mode-trigger" ref={menuRef}>
          <button
            type="button"
            className="chat__mode-button"
            onClick={() => setModeMenuOpen((open) => !open)}
            aria-label="Select chat mode"
          >
            {formatMode(mode)}
            <span className="chat__mode-caret">▾</span>
          </button>
          {modeMenuOpen ? (
            <div className="chat__mode-menu">
              {MODES.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={option === mode ? "chat__mode-option--active" : undefined}
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
        <button className="chat__send-button" type="submit" aria-label="Send" disabled={isSending}>
          ➤
        </button>
      </form>
    </article>
  );
};

export default ChatPanel;
