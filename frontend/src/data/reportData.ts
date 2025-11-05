export type SourceScores = {
  credibility: number;
  validity: number;
  accuracy?: number;
  overall?: number;
};

export type InternalSource = {
  id: string;
  name: string;
  filePath: string;
  isLocal?: boolean;
  summary?: string;
  scores?: SourceScores;
};

export const internalSources: InternalSource[] = [
  {
    id: "2025-world-hunger",
    name: "2025 World Hunger.pdf",
    filePath: "/example_sources/2025_world_hunger.pdf",
    summary:
      "Global Hunger Index review outlining progress toward zero hunger, highlighting regional disparities, chronic food insecurity drivers, and resilience initiatives for 2025.",
    scores: {
      credibility: 0.83,
      validity: 0.72,
      overall: 0.78
    }
  },
  {
    id: "supply-chain-disruptions",
    name: "Disruptions in the Food Supply Chain.pdf",
    filePath: "/example_sources/disruptions_in_the_food_supply_chain.pdf",
    summary:
      "Analysis of bottlenecks across production, storage, and logistics networks with recommendations on restoring food chain continuity after major shocks.",
    scores: {
      credibility: 0.78,
      validity: 0.66,
      overall: 0.72
    }
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

export const reportDocument = {
  id: "world-hunger-disruptions-report",
  name: "World Hunger & Food Chain Disruptions Report.pdf",
  filePath: "/example_sources/World_Hunger_and_Food_Chain_Disruptions_Report.pdf"
};
