export type InternalSource = {
  id: string;
  name: string;
  filePath: string;
  isLocal?: boolean;
};

export const internalSources: InternalSource[] = [
  {
    id: "2025-world-hunger",
    name: "2025 World Hunger.pdf",
    filePath: "/example_sources/2025_world_hunger.pdf"
  },
  {
    id: "supply-chain-disruptions",
    name: "Disruptions in the Food Supply Chain.pdf",
    filePath: "/example_sources/disruptions_in_the_food_supply_chain.pdf"
  }
];

export const recommendedSources = [
  "Global Food Resilience Index 2025",
  "Nutrition Equity Observatory Brief",
  "AgriSupply Chain Stability Outlook",
  "Climate Resilient Harvests 2024",
  "Urban Food Access Benchmark 2025",
  "FAO Logistics Pulse - June 2025"
];

export const reportSummary =
  "The 2025 World Hunger and Food Chain Disruptions report highlights how climate shocks, conflict-driven displacement, " +
  "and fragile logistics networks are converging to keep 735 million people in chronic food insecurity. It contrasts " +
  "regions with resilient storage and cold-chain investments against those relying on volatile grain imports, and " +
  "underscores that rapid response funds and nutrition-focused safety nets remain under-capitalised.";

export const reportScores = {
  overall: 0.78,
  accuracy: 0.74,
  credibility: 0.81,
  validity: 0.69
};

export const chatMessages = [
  { id: 1, author: "System", text: "Welcome back! Ask anything about improving this report." },
  { id: 2, author: "User", text: "What sections should I revise first?" }
];
