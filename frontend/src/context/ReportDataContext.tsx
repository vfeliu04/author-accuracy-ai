import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  useEffect,
  type ReactNode
} from "react";
import type { InternalSource } from "../data/reportData";
import {
  uploadSources,
  uploadReport,
  fetchUploads,
  deleteUpload,
  type UploadRecord,
  type ReportSummaryResponse,
  type ChatHistoryEntry,
  fetchJob
} from "../api/client";

type ReportDataContextValue = {
  internalSources: InternalSource[];
  reportDocument: InternalSource | null;
  addInternalSource: (file: File) => Promise<void>;
  removeInternalSource: (id: string) => Promise<void>;
  setReportDocument: (file: File) => Promise<void>;
  getInternalSourceById: (id: string) => InternalSource | undefined;
  refreshUploads: () => Promise<void>;
  summaryData: ReportSummaryResponse | null;
  setSummaryData: (summary: ReportSummaryResponse | null) => void;
  getChatMessages: (jobId: string) => ChatHistoryEntry[];
  setChatMessages: (jobId: string, messages: ChatHistoryEntry[]) => void;
  jobStatus: JobStatus | null;
  setJobStatus: (status: JobStatus | null) => void;
  refreshJobStatus: (jobId: string) => Promise<void>;
};

type JobStatus = {
  job_id: string;
  status: string;
  updated_at?: string;
};

const ReportDataContext = createContext<ReportDataContextValue | undefined>(undefined);

const MAX_DISPLAY_LENGTH = 32;

const toTitleCase = (value: string) =>
  value
    .split(" ")
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1).toLowerCase())
    .join(" ");

const formatDisplayName = (fileName: string) => {
  const extensionMatch = fileName.match(/\.[^/.]+$/);
  const extension = extensionMatch ? extensionMatch[0] : "";
  const baseRaw = extension ? fileName.slice(0, -extension.length) : fileName;
  const normalizedBase = toTitleCase(baseRaw.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim());

  if (!extension) {
    return normalizedBase;
  }

  let candidate = `${normalizedBase}${extension}`;
  if (candidate.length <= MAX_DISPLAY_LENGTH) {
    return candidate;
  }

  const allowedBaseLength = Math.max(MAX_DISPLAY_LENGTH - extension.length - 3, 8);
  const truncated = normalizedBase.slice(0, allowedBaseLength).trimEnd();
  return `${truncated}...${extension}`;
};

type ReportDataProviderProps = {
  children: ReactNode;
};

export const ReportDataProvider = ({ children }: ReportDataProviderProps) => {
  const [sources, setSources] = useState<InternalSource[]>([]);
  const [reportDoc, setReportDoc] = useState<InternalSource | null>(null);
  const loadingRef = useRef(false);
  const [summaryData, setSummaryDataState] = useState<ReportSummaryResponse | null>(null);
  const [chatHistoryMap, setChatHistoryMap] = useState<Record<string, ChatHistoryEntry[]>>({});
  const [jobStatus, setJobStatusState] = useState<JobStatus | null>(null);

  const fromUploadRecord = useCallback((record: UploadRecord): InternalSource => {
    const displayName = formatDisplayName(record.file_name);
    return {
      id: record.upload_id,
      name: displayName,
      filePath: record.file_url,
      isLocal: false,
      summary: undefined,
      scores: undefined
    };
  }, []);

  const refreshUploads = useCallback(async () => {
    if (loadingRef.current) {
      return;
    }
    loadingRef.current = true;
    try {
      const [sourceUploads, reportUploads] = await Promise.all([
        fetchUploads("SOURCE"),
        fetchUploads("REPORT")
      ]);
      setSources(sourceUploads.map(fromUploadRecord));
      setReportDoc(reportUploads[0] ? fromUploadRecord(reportUploads[0]) : null);
    } finally {
      loadingRef.current = false;
    }
  }, [fromUploadRecord]);

  useEffect(() => {
    refreshUploads().catch(() => undefined);
  }, [refreshUploads]);

  const addInternalSource = useCallback(async (file: File) => {
    const uploads = await uploadSources([file]);
    setSources((prev) => [...prev, ...uploads.map(fromUploadRecord)]);
  }, [fromUploadRecord]);

  const removeInternalSource = useCallback(
    async (id: string) => {
      await deleteUpload(id);
      setSources((prev) => prev.filter((source) => source.id !== id));
    },
    []
  );

  const setReportDocument = useCallback(async (file: File) => {
    const upload = await uploadReport(file);
    setReportDoc(fromUploadRecord(upload));
  }, [fromUploadRecord]);

  const getInternalSourceById = useCallback(
    (id: string) => sources.find((source) => source.id === id),
    [sources]
  );

  const setSummaryData = useCallback(
    (summary: ReportSummaryResponse | null) => {
      setSummaryDataState(summary);
      if (summary) {
        setReportDoc({
          id: summary.report.id,
          name: summary.report.name,
          filePath: summary.report.pdf_url ?? "",
          isLocal: false,
          summary: summary.report.summary
        });
      }
    },
    []
  );

  const getChatMessages = useCallback(
    (jobId: string) => chatHistoryMap[jobId] ?? [],
    [chatHistoryMap]
  );

  const setChatMessages = useCallback((jobId: string, messages: ChatHistoryEntry[]) => {
    setChatHistoryMap((prev) => ({ ...prev, [jobId]: messages }));
  }, []);

  const setJobStatus = useCallback((status: JobStatus | null) => {
    setJobStatusState(status);
  }, []);

  const refreshJobStatus = useCallback(
    async (jobId: string) => {
      try {
        const job = await fetchJob(jobId);
        setJobStatusState({ job_id: job.job_id, status: job.status, updated_at: job.updated_at });
        if (job.status === "DONE") {
          localStorage.setItem("active_job_id", job.job_id);
        }
      } catch (error) {
        console.error(error);
      }
    },
    []
  );

  const value = useMemo(
    () => ({
      internalSources: sources,
      reportDocument: reportDoc,
      addInternalSource,
      removeInternalSource,
      setReportDocument,
      getInternalSourceById,
      refreshUploads,
      summaryData,
      setSummaryData,
      getChatMessages,
      setChatMessages,
      jobStatus,
      setJobStatus,
      refreshJobStatus
    }),
    [
      sources,
      reportDoc,
      addInternalSource,
      removeInternalSource,
      setReportDocument,
      getInternalSourceById,
      refreshUploads,
      summaryData,
      setSummaryData,
      chatHistoryMap,
      getChatMessages,
      setChatMessages,
      jobStatus,
      setJobStatus,
      refreshJobStatus
    ]
  );

  return <ReportDataContext.Provider value={value}>{children}</ReportDataContext.Provider>;
};

export const useReportData = () => {
  const context = useContext(ReportDataContext);
  if (!context) {
    throw new Error("useReportData must be used within a ReportDataProvider");
  }
  return context;
};
