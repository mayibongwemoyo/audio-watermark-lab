
import { createContext, useState, ReactNode } from "react";
// import { WatermarkEmbedResult } from "@/services/api";
import { WatermarkResultsContextType } from "@/services/api"; // <-- Import the new name


export interface WatermarkResults {
  snr_db: number;
  ber: number;
  detection_probability: number;
  processed_audio_url: string;
  method: string;
  // step_results: WatermarkEmbedResult[];
  results: WatermarkResultsContextType ;
}

interface ResultsContextType {
  results: WatermarkResultsContextType | null; // Use the new type
  setResults: (results: WatermarkResultsContextType | null) => void; // Use the new type
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
  const [results, setResults] = useState<WatermarkResultsContextType | null>(null);

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
