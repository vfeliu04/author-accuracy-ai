import { Route, Routes } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import ReportDashboard from "./components/ReportDashboard";
import SourceDetail from "./components/SourceDetail";
import ReportDetail from "./components/ReportDetail";
import RecommendedSourceDetail from "./components/RecommendedSourceDetail";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/dashboard" element={<ReportDashboard />} />
      <Route path="/dashboard/sources/:sourceId" element={<SourceDetail />} />
      <Route path="/dashboard/report" element={<ReportDetail />} />
      <Route path="/dashboard/recommendations/:sourceIndex" element={<RecommendedSourceDetail />} />
    </Routes>
  );
};

export default App;
