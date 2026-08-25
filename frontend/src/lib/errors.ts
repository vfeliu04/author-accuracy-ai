// Raw exception text is for logs; users get the translation (the raw message
// stays visible in a collapsible block wherever this is rendered).
const ERROR_HINTS: Array<{ match: RegExp; hint: string }> = [
  {
    match: /APIConnectionError|Connection error/i,
    hint: "The server lost its network connection mid-run (laptop sleep or dropped Wi-Fi are the usual causes). Retrying resumes where it stopped."
  },
  {
    match: /TimeoutError.*Batch|still 'in_progress'/i,
    hint: "The verification batch was still queued on the provider's side when the app stopped waiting. The batch keeps its place — retrying reattaches to it at no extra cost."
  },
  {
    match: /credit balance|billing/i,
    hint: "The API account looks out of credit — top up at console.anthropic.com, then retry."
  }
];

export function humanizeError(error: string | null): string | null {
  if (!error) return null;
  return ERROR_HINTS.find(({ match }) => match.test(error))?.hint ?? null;
}
