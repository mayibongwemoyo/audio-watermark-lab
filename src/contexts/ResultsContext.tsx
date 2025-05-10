
import { createContext, useState, ReactNode } from "react";
import { WatermarkEmbedResult } from "@/services/api";

export interface WatermarkResults {
  snr_db: number;
  ber: number;
  detection_probability: number;
  processed_audio_url: string;
  method: string;
  step_results: WatermarkEmbedResult[];
}

interface ResultsContextType {
  results: WatermarkResults | null;
  setResults: (results: WatermarkResults | null) => void;
  clearResults: () => void;
}

export const ResultsContext = createContext<ResultsContextType>({
  results: null,
  setResults: () => {},
  clearResults: () => {},
});

interface ResultsProviderProps {
  children: ReactNode;
}

export const ResultsProvider = ({ children }: ResultsProviderProps) => {
  const [results, setResults] = useState<WatermarkResults | null>(null);

  const clearResults = () => {
    setResults(null);
  };

  return (
    <ResultsContext.Provider value={{ 
      results, 
      setResults,
      clearResults
    }}>
      {children}
    </ResultsContext.Provider>
  );
};
