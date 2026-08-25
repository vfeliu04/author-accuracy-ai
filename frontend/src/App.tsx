import { Navigate, Route, Routes } from "react-router-dom";
import HomePage from "./components/HomePage";
import ReportDashboard from "./components/ReportDashboard";
import ClaimsWorkspace from "./components/ClaimsWorkspace";
import ComparePage from "./components/ComparePage";
import SourceDetail from "./components/SourceDetail";
import ReportDetail from "./components/ReportDetail";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/runs/:runId" element={<ReportDashboard />} />
      <Route path="/runs/:runId/report" element={<ReportDetail />} />
      <Route path="/runs/:runId/sources/:sourceId" element={<SourceDetail />} />
      <Route path="/runs/:runId/workspace" element={<ClaimsWorkspace />} />
      <Route path="/compare" element={<ComparePage />} />
      {/* Legacy entry points all land on the gallery. */}
      <Route path="/runs" element={<Navigate to="/" replace />} />
      <Route path="/dashboard" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

export default App;
