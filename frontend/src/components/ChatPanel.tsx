import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type ChatMessage = {
  id: number;
  author: string;
  text: string;
};

type ChatPanelProps = {
  messages: ChatMessage[];
  onSendMessage: (text: string) => Promise<void>;
  isSending?: boolean;
};

// ChatPanel mimics the conversational area for discussing improvements to the report.
const ChatPanel = ({ messages, onSendMessage, isSending = false }: ChatPanelProps) => {
  const [draft, setDraft] = useState("");
  const historyRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = historyRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages, isSending]);

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
    <article className="card card--chat">
      <header className="card__header card__header--chat">
        <h2>Chat</h2>
      </header>
      <div className="chat__history" ref={historyRef}>
        {messages.map((message) => {
          const isSystem = message.author.toLowerCase() === "system";
          return (
            <div
              key={message.id}
              className={`chat__message ${
                isSystem ? "chat__message--system" : "chat__message--user"
              } ${isSystem ? "chat__message--left" : "chat__message--right"}`}
            >
              <span className="chat__message-author">{message.author}</span>
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
      <form className="chat__composer" onSubmit={handleSubmit}>
        <input
          className="chat__input"
          type="text"
          placeholder="Type your question..."
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          disabled={isSending}
        />
        <button className="chat__send-button" type="submit" aria-label="Send" disabled={isSending}>
          ➤
        </button>
      </form>
    </article>
  );
};

export default ChatPanel;
