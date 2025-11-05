import { Route, Routes } from "react-router-dom";
import ReportDashboard from "./components/ReportDashboard";
import SourceDetail from "./components/SourceDetail";
import ReportDetail from "./components/ReportDetail";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<ReportDashboard />} />
      <Route path="/sources/:sourceId" element={<SourceDetail />} />
      <Route path="/report" element={<ReportDetail />} />
    </Routes>
  );
};

export default App;
