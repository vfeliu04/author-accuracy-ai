import { Navigate, Route, Routes } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import HistoryPage from "./components/HistoryPage";
import ReportDashboard from "./components/ReportDashboard";
import ClaimsWorkspace from "./components/ClaimsWorkspace";
import ComparePage from "./components/ComparePage";
import SourceDetail from "./components/SourceDetail";
import ReportDetail from "./components/ReportDetail";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/runs" element={<HistoryPage />} />
      <Route path="/runs/:runId" element={<ReportDashboard />} />
      <Route path="/runs/:runId/report" element={<ReportDetail />} />
      <Route path="/runs/:runId/sources/:sourceId" element={<SourceDetail />} />
      <Route path="/runs/:runId/workspace" element={<ClaimsWorkspace />} />
      <Route path="/compare" element={<ComparePage />} />
      {/* The old single-report dashboard is now the run history. */}
      <Route path="/dashboard" element={<Navigate to="/runs" replace />} />
    </Routes>
  );
};

export default App;
