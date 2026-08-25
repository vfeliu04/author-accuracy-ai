// Deterministic, topic-neutral card icon: the same run always renders the
// same emoji, with no backend involvement — seeded by title (or id).
const EMOJIS = ["📄", "📊", "📘", "🗂️", "🔍", "📈", "📚", "📝", "🧾", "🧭"];

export function emojiFor(seed: string): string {
  let hash = 0;
  for (let i = 0; i < seed.length; i += 1) {
    hash = (hash * 31 + seed.charCodeAt(i)) | 0;
  }
  return EMOJIS[Math.abs(hash) % EMOJIS.length];
}
