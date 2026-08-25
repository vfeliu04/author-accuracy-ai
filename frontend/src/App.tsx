import { Navigate, Route, Routes, useParams } from "react-router-dom";
import HomePage from "./components/HomePage";
import RunView from "./components/RunView";
import ComparePage from "./components/ComparePage";

// Pre-redesign bookmarks land on the equivalent focus mode.
function LegacyRunRedirect({ focus }: { focus: string }) {
  const { runId, sourceId } = useParams<{ runId: string; sourceId?: string }>();
  const suffix = sourceId ? `&source=${sourceId}` : "";
  return <Navigate to={`/runs/${runId}?focus=${focus}${suffix}`} replace />;
}

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/runs/:runId" element={<RunView />} />
      <Route path="/compare" element={<ComparePage />} />
      {/* Legacy entry points. */}
      <Route path="/runs/:runId/workspace" element={<LegacyRunRedirect focus="claims" />} />
      <Route path="/runs/:runId/report" element={<LegacyRunRedirect focus="report" />} />
      <Route
        path="/runs/:runId/sources/:sourceId"
        element={<LegacyRunRedirect focus="credibility" />}
      />
      <Route path="/runs" element={<Navigate to="/" replace />} />
      <Route path="/dashboard" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

export default App;
