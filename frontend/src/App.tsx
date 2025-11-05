import { Route, Routes } from "react-router-dom";
import ReportDashboard from "./components/ReportDashboard";
import SourceDetail from "./components/SourceDetail";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<ReportDashboard />} />
      <Route path="/sources/:sourceId" element={<SourceDetail />} />
    </Routes>
  );
};

export default App;
