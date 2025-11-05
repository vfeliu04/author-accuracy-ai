import { Route, Routes } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import ReportDashboard from "./components/ReportDashboard";
import SourceDetail from "./components/SourceDetail";
import ReportDetail from "./components/ReportDetail";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/dashboard" element={<ReportDashboard />} />
      <Route path="/dashboard/sources/:sourceId" element={<SourceDetail />} />
      <Route path="/dashboard/report" element={<ReportDetail />} />
    </Routes>
  );
};

export default App;
