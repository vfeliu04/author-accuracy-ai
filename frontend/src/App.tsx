import { Navigate, Route, Routes } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import HistoryPage from "./components/HistoryPage";
import ReportDashboard from "./components/ReportDashboard";
import SourceDetail from "./components/SourceDetail";
import ReportDetail from "./components/ReportDetail";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/runs" element={<HistoryPage />} />
      <Route path="/runs/:runId" element={<ReportDashboard />} />
      {/* Legacy detail routes — re-pointed at the v2 shapes in F5. */}
      <Route path="/dashboard/sources/:sourceId" element={<SourceDetail />} />
      <Route path="/dashboard/report" element={<ReportDetail />} />
      {/* The old single-report dashboard is now the run history. */}
      <Route path="/dashboard" element={<Navigate to="/runs" replace />} />
    </Routes>
  );
};

export default App;
