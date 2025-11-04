import InternalSourcesPanel from "./InternalSourcesPanel";
import SummaryPanel from "./SummaryPanel";
import ChatPanel from "./ChatPanel";
import RatingPanel from "./RatingPanel";
import RecommendedSourcesPanel from "./RecommendedSourcesPanel";

// ReportDashboard represents the full page layout for the report quality view.
const ReportDashboard = () => {
  const internalSources = [
    "Source One",
    "Source Two",
    "Source Three",
    "Source Four",
    "Source Five",
    "Source Six",
    "Source Seven",
    "Source Eight",
    "Source Nine",
    "Source Ten"
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
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ligula erat, molestie vel tortor at, aliquet imperdiet mi. " +
    "Donec posuere interdum mi vitae fermentum.";

  const recommendedSources = [
    "Source One",
    "Source Two",
    "Source Three",
    "Source Four",
    "Source Five",
    "Source Six",
    "Source Seven",
    "Source Eight",
    "Source Nine",
    "Source Ten"
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
