import { useNavigate } from "react-router-dom";
import InternalSourcesPanel from "./InternalSourcesPanel";
import SummaryPanel from "./SummaryPanel";
import ChatPanel from "./ChatPanel";
import RatingPanel from "./RatingPanel";
import RecommendedSourcesPanel from "./RecommendedSourcesPanel";
import {
  chatMessages,
  recommendedSources,
  reportDocument,
  reportScores,
  reportSummary
} from "../data/reportData";
import { useReportData } from "../context/ReportDataContext";

// ReportDashboard represents the full page layout for the report quality view.
const ReportDashboard = () => {
  const navigate = useNavigate();
  const { internalSources, addInternalSource } = useReportData();

  const handleSourceSelect = (sourceId: string) => {
    navigate(`/sources/${sourceId}`);
  };

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <h1>World - Hunger Report</h1>
      </header>
      <main className="dashboard__content">
        <section className="dashboard__column dashboard__column--left dashboard__column--stacked-left">
          <InternalSourcesPanel
            sources={internalSources}
            onSelectSource={handleSourceSelect}
            onAddSource={addInternalSource}
          />
          <SummaryPanel
            summary={reportSummary}
            reportLabel={reportDocument.name}
            onOpenReport={() => navigate(`/report`)}
          />
        </section>
        <section className="dashboard__column dashboard__column--center">
          <ChatPanel messages={chatMessages} />
        </section>
        <section className="dashboard__column dashboard__column--right dashboard__column--stacked-right">
          <RatingPanel scores={reportScores} showAccuracy />
          <RecommendedSourcesPanel sources={recommendedSources} />
        </section>
      </main>
    </div>
  );
};

export default ReportDashboard;
