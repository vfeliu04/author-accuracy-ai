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
import { internalSources as initialInternalSources } from "../data/reportData";

type ReportDataContextValue = {
  internalSources: InternalSource[];
  addInternalSource: (file: File) => void;
  getInternalSourceById: (id: string) => InternalSource | undefined;
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

const generateSourceId = (fileName: string) => {
  const baseRaw = fileName.replace(/\.[^/.]+$/, "");
  const slug =
    baseRaw
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "source";
  return `${slug}-${Date.now()}`;
};

type ReportDataProviderProps = {
  children: ReactNode;
};

export const ReportDataProvider = ({ children }: ReportDataProviderProps) => {
  const [sources, setSources] = useState<InternalSource[]>(initialInternalSources);
  const localUrlsRef = useRef<string[]>([]);

  const addInternalSource = useCallback((file: File) => {
    const displayName = formatDisplayName(file.name);
    const sourceId = generateSourceId(file.name);
    const objectUrl = URL.createObjectURL(file);

    localUrlsRef.current.push(objectUrl);

    setSources((prev) => [
      ...prev,
      {
        id: sourceId,
        name: displayName,
        filePath: objectUrl,
        isLocal: true
      }
    ]);
  }, []);

  const getInternalSourceById = useCallback(
    (id: string) => sources.find((source) => source.id === id),
    [sources]
  );

  useEffect(
    () => () => {
      localUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    },
    []
  );

  const value = useMemo(
    () => ({
      internalSources: sources,
      addInternalSource,
      getInternalSourceById
    }),
    [sources, addInternalSource, getInternalSourceById]
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
