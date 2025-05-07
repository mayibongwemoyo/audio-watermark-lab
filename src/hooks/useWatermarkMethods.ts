
import { useState, useEffect } from "react";
import { api } from "@/services/api";

interface WatermarkMethod {
  name: string;
  description: string;
  available: boolean;
}

export function useWatermarkMethods() {
  const [methods, setMethods] = useState<Record<string, WatermarkMethod>>({});
  const [audioSealAvailable, setAudioSealAvailable] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchMethods = async () => {
      try {
        setIsLoading(true);
        const data = await api.getMethods();
        setMethods(data.methods);
        setAudioSealAvailable(data.audioseal_available);
        setError(null);
      } catch (err) {
        setError("Failed to fetch watermarking methods");
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchMethods();
  }, []);
  
  return {
    methods,
    audioSealAvailable,
    isLoading,
    error
  };
}
