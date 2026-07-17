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
  usageCount?: number;
};

export const internalSources: InternalSource[] = [];
import type { RecommendedSource } from "../api/client";

export const recommendedSources: RecommendedSource[] = [];
export const reportSummary = "";
export const reportScores = {
  overall: 0,
  accuracy: 0,
  credibility: 0,
  validity: 0
};
export const chatMessages: { id: number; author: string; text: string }[] = [];
export const reportDocument = null;
