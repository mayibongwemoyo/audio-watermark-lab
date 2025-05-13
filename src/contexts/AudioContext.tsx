
import { createContext, useState, ReactNode } from "react";

interface AudioContextType {
  audioFile: File | null;
  audioUrl: string | null;
  audioName: string | null;
  setAudioFile: (file: File | null) => void;
  setAudioUrl: (url: string | null) => void;
  setAudioName: (name: string | null) => void;
  clearAudio: () => void;
}

export const AudioContext = createContext<AudioContextType>({
  audioFile: null,
  audioUrl: null,
  audioName: null,
  setAudioFile: () => {},
  setAudioUrl: () => {},
  setAudioName: () => {},
  clearAudio: () => {},
});

interface AudioProviderProps {
  children: ReactNode;
}

export const AudioProvider = ({ children }: AudioProviderProps) => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioName, setAudioName] = useState<string | null>(null);

  const clearAudio = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioFile(null);
    setAudioUrl(null);
    setAudioName(null);
  };

  return (
    <AudioContext.Provider value={{ 
      audioFile, 
      audioUrl, 
      audioName,
      setAudioFile, 
      setAudioUrl,
      setAudioName,
      clearAudio 
    }}>
      {children}
    </AudioContext.Provider>
  );
};
