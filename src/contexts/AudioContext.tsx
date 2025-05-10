
import { createContext, useState, ReactNode } from "react";

interface AudioContextType {
  audioFile: File | null;
  audioUrl: string | null;
  setAudioFile: (file: File | null) => void;
  setAudioUrl: (url: string | null) => void;
  clearAudio: () => void;
}

export const AudioContext = createContext<AudioContextType>({
  audioFile: null,
  audioUrl: null,
  setAudioFile: () => {},
  setAudioUrl: () => {},
  clearAudio: () => {},
});

interface AudioProviderProps {
  children: ReactNode;
}

export const AudioProvider = ({ children }: AudioProviderProps) => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const clearAudio = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioFile(null);
    setAudioUrl(null);
  };

  return (
    <AudioContext.Provider value={{ 
      audioFile, 
      audioUrl, 
      setAudioFile, 
      setAudioUrl,
      clearAudio 
    }}>
      {children}
    </AudioContext.Provider>
  );
};
