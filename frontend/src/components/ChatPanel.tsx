import { FormEvent, useState } from "react";

type ChatMessage = {
  id: number;
  author: string;
  text: string;
};

type ChatPanelProps = {
  messages: ChatMessage[];
};

// ChatPanel mimics the conversational area for discussing improvements to the report.
const ChatPanel = ({ messages: initialMessages }: ChatPanelProps) => {
  const [messages, setMessages] = useState(initialMessages);
  const [draft, setDraft] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmed = draft.trim();
    if (!trimmed) {
      return;
    }

    const timestamp = Date.now();
    setMessages((prev) => [
      ...prev,
      { id: timestamp, author: "User", text: trimmed },
      { id: timestamp + 1, author: "System", text: "Chat function not working at the moment." }
    ]);
    setDraft("");
  };

  return (
    <article className="card card--chat">
      <header className="card__header card__header--chat">
        <h2>Chat</h2>
      </header>
      <div className="chat__history">
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
              <p className="chat__message-text">{message.text}</p>
            </div>
          );
        })}
      </div>
      <form className="chat__composer" onSubmit={handleSubmit}>
        <input
          className="chat__input"
          type="text"
          placeholder="Type your question..."
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button className="chat__send-button" type="submit" aria-label="Send">
          ➤
        </button>
      </form>
    </article>
  );
};

export default ChatPanel;
