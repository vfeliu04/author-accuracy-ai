import InternalSourcesPanel from "./InternalSourcesPanel";
import SummaryPanel from "./SummaryPanel";
import ChatPanel from "./ChatPanel";
import RatingPanel from "./RatingPanel";
import RecommendedSourcesPanel from "./RecommendedSourcesPanel";

// ReportDashboard represents the full page layout for the report quality view.
const ReportDashboard = () => {
  const internalSources = [
    "2025 World Hunger.pdf",
    "World Hunger & Food Chain Disruptions.pdf",
    "Disruptions in the Food Supply Chain.pdf"
  ];

  const scores = {
    overall: 0.78,
    accuracy: 0.74,
    credibility: 0.81,
    validity: 0.69
  };

  const chatMessages = [
    { id: 1, author: "System", text: "Welcome back! Ask anything about improving this report." },
    { id: 2, author: "User", text: "What sections should I revise first?" }
  ];

  const summaryText =
    "The 2025 World Hunger and Food Chain Disruptions report highlights how climate shocks, conflict-driven displacement, " +
    "and fragile logistics networks are converging to keep 735 million people in chronic food insecurity. It contrasts " +
    "regions with resilient storage and cold-chain investments against those relying on volatile grain imports, and " +
    "underscores that rapid response funds and nutrition-focused safety nets remain under-capitalised.";

  const recommendedSources = [
    "Global Food Resilience Index 2025",
    "Nutrition Equity Observatory Brief",
    "AgriSupply Chain Stability Outlook",
    "Climate Resilient Harvests 2024",
    "Urban Food Access Benchmark 2025",
    "FAO Logistics Pulse - June 2025"
  ];

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <h1>World - Hunger Report</h1>
      </header>
      <main className="dashboard__content">
        <section className="dashboard__column dashboard__column--left dashboard__column--stacked-left">
          <InternalSourcesPanel sources={internalSources} />
          <SummaryPanel summary={summaryText} />
        </section>
        <section className="dashboard__column dashboard__column--center">
          <ChatPanel messages={chatMessages} />
        </section>
        <section className="dashboard__column dashboard__column--right dashboard__column--stacked-right">
          <RatingPanel scores={scores} />
          <RecommendedSourcesPanel sources={recommendedSources} />
        </section>
      </main>
    </div>
  );
};

export default ReportDashboard;
