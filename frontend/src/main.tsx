import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";
import { BrowserRouter } from "react-router-dom";
import { ReportDataProvider } from "./context/ReportDataContext";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <ReportDataProvider>
        <App />
      </ReportDataProvider>
    </BrowserRouter>
  </React.StrictMode>
);
