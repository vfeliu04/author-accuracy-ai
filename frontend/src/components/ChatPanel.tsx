import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMode } from "../api/client";

export type ChatMessage = {
  id: number;
  author: string;
  text: string;
};

type ChatPanelProps = {
  messages: ChatMessage[];
  onSendMessage: (text: string) => Promise<void>;
  isSending?: boolean;
  mode: ChatMode;
  modeLocked: boolean;
  onModeChange: (mode: ChatMode) => void;
  onModeReset: () => void;
  modeSuggestion?: ChatMode | null;
  onSuggestionAccept?: (mode: ChatMode) => void;
  onSuggestionDismiss?: () => void;
};

// ChatPanel mimics the conversational area for discussing improvements to the report.
const ChatPanel = ({
  messages,
  onSendMessage,
  isSending = false,
  mode,
  modeLocked,
  onModeChange,
  onModeReset,
  modeSuggestion,
  onSuggestionAccept,
  onSuggestionDismiss
}: ChatPanelProps) => {
  const [draft, setDraft] = useState("");
  const historyRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
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
      if (!modeMenuOpen) {
        return;
      }
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

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmed = draft.trim();
    if (!trimmed) {
      return;
    }

    setDraft("");
    try {
      await onSendMessage(trimmed);
    } catch (error) {
      // swallow errors since parent already handles UI feedback
    }
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
          const isUser = message.author.toLowerCase() === "user";
          const displayAuthor = isUser ? "User" : "author.ai";
          return (
            <div
              key={message.id}
              className={`chat__message ${
                isUser ? "chat__message--user" : "chat__message--system"
              } ${isUser ? "chat__message--right" : "chat__message--left"}`}
            >
              <span className="chat__message-author">{displayAuthor}</span>
              <div className="chat__message-text">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.text}
                </ReactMarkdown>
              </div>
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
      <form className="chat__composer-inner" onSubmit={handleSubmit}>
          <input
            className="chat__input"
            type="text"
            placeholder="Ask about your report..."
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={isSending}
          />
          <div className="chat__mode-trigger" ref={menuRef}>
            <button
              type="button"
              className="chat__mode-button"
              onClick={() => setModeMenuOpen((open) => !open)}
              aria-label="Select chat mode"
            >
              {modeLocked ? formatMode(mode) : `Auto · ${formatMode(mode)}`}
              <span className="chat__mode-caret">▾</span>
            </button>
            {modeMenuOpen ? (
              <div className="chat__mode-menu">
                <button
                  type="button"
                  onClick={() => {
                    onModeReset();
                    setModeMenuOpen(false);
                  }}
                >
                  Auto (let assistant choose)
                </button>
                <hr />
                {(["evidence", "guidance", "creative"] as ChatMode[]).map((option) => (
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
      {modeSuggestion ? (
        <div className="chat__mode-suggestion">
          <span>{`Try ${modeSuggestion} mode for better results.`}</span>
          <div className="chat__mode-suggestion-actions">
            <button type="button" onClick={() => onSuggestionAccept?.(modeSuggestion!)}>
              Switch
            </button>
            <button type="button" onClick={onSuggestionDismiss}>
              Dismiss
            </button>
          </div>
        </div>
      ) : null}
    </article>
  );
};

export default ChatPanel;
