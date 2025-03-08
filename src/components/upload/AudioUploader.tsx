
import { useState, useRef } from "react";
import { Upload, Mic, X, Play, Pause, Volume2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";

const AudioUploader = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    handleFiles(files);
  };
  
  const handleFiles = (files: FileList) => {
    if (files.length === 0) return;
    
    const file = files[0];
    if (!file.type.startsWith("audio/")) {
      toast.error("Please upload an audio file");
      return;
    }
    
    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    toast.success("Audio file uploaded successfully");
  };
  
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };
  
  const clearAudio = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioFile(null);
    setAudioUrl(null);
    setIsPlaying(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };
  
  const togglePlayPause = () => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play()
        .catch(error => {
          console.error("Playback error:", error);
          toast.error("Failed to play audio");
        });
    }
    setIsPlaying(!isPlaying);
  };
  
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        chunksRef.current = [];
        
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        
        // Convert blob to file
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        setAudioFile(file);
        
        toast.success("Recording saved successfully");
      };
      
      chunksRef.current = [];
      mediaRecorder.start();
      setIsRecording(true);
      toast.info("Recording started...");
    } catch (error) {
      console.error("Recording error:", error);
      toast.error("Failed to access microphone");
    }
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      // Stop all audio tracks
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };
  
  return (
    <section id="upload" className="py-24 px-6 md:px-10">
      <div className="max-w-4xl mx-auto">
        <div className="flex flex-col gap-3 mb-12 animate-fade-in">
          <div className="flex items-center gap-2">
            <div className="h-8 w-1 bg-black dark:bg-white rounded-full"></div>
            <h2 className="text-xl md:text-2xl font-semibold">Upload Audio</h2>
          </div>
          <p className="text-black/70 dark:text-white/70 ml-3 pl-3">
            Upload an audio file or record directly to begin the watermarking process
          </p>
        </div>
        
        {!audioFile ? (
          <div 
            className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center min-h-[300px] transition-all duration-300 animate-fade-in ${
              isDragging 
                ? "border-black bg-black/5 dark:border-white dark:bg-white/5" 
                : "border-black/20 dark:border-white/20 hover:border-black/50 dark:hover:border-white/50"
            }`}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input 
              type="file" 
              accept="audio/*" 
              className="hidden" 
              onChange={handleFileInputChange}
              id="audio-input"
            />
            
            <Upload size={40} className="mb-4 text-black/40 dark:text-white/40" />
            <h3 className="text-xl font-medium mb-2">Drag & drop audio file here</h3>
            <p className="text-black/60 dark:text-white/60 mb-6 text-center max-w-md">
              Or use one of the options below to add your audio
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button 
                variant="outline"
                className="rounded-full px-6"
                onClick={() => document.getElementById("audio-input")?.click()}
              >
                <Upload size={16} className="mr-2" />
                Browse Files
              </Button>
              
              {!isRecording ? (
                <Button 
                  className="rounded-full px-6 bg-black hover:bg-black/90 text-white dark:bg-white dark:text-black dark:hover:bg-white/90"
                  onClick={startRecording}
                >
                  <Mic size={16} className="mr-2" />
                  Record Audio
                </Button>
              ) : (
                <Button 
                  variant="destructive"
                  className="rounded-full px-6"
                  onClick={stopRecording}
                >
                  <X size={16} className="mr-2" />
                  Stop Recording
                </Button>
              )}
            </div>
          </div>
        ) : (
          <Card className="rounded-xl overflow-hidden animate-scale-in">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Volume2 size={20} />
                  <div>
                    <h3 className="font-medium">{audioFile.name}</h3>
                    <p className="text-xs text-black/60 dark:text-white/60">
                      {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={clearAudio}
                  className="rounded-full text-black/60 hover:text-black dark:text-white/60 dark:hover:text-white"
                >
                  <X size={18} />
                </Button>
              </div>
              
              <div className="bg-black/5 dark:bg-white/5 rounded-lg p-4 flex items-center gap-4">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={togglePlayPause}
                  className="rounded-full h-10 w-10"
                >
                  {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                </Button>
                
                <div className="w-full">
                  <div className="w-full h-1 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full w-0 bg-black dark:bg-white rounded-full transition-all duration-100" id="audio-progress"></div>
                  </div>
                </div>
              </div>
              
              {audioUrl && (
                <audio 
                  ref={audioRef} 
                  src={audioUrl}
                  className="hidden"
                  onEnded={() => setIsPlaying(false)}
                  onTimeUpdate={() => {
                    if (audioRef.current) {
                      const progress = (audioRef.current.currentTime / audioRef.current.duration) * 100;
                      const progressElement = document.getElementById("audio-progress");
                      if (progressElement) {
                        progressElement.style.width = `${progress}%`;
                      }
                    }
                  }}
                />
              )}
            </div>
            
            <div className="px-6 pb-6">
              <Button 
                className="w-full rounded-full bg-black hover:bg-black/90 text-white dark:bg-white dark:text-black dark:hover:bg-white/90"
                onClick={() => {
                  toast.success("Proceeding to watermarking...");
                  document.getElementById('watermark')?.scrollIntoView({ behavior: 'smooth' });
                }}
              >
                Proceed to Watermarking
              </Button>
            </div>
          </Card>
        )}
      </div>
    </section>
  );
};

export default AudioUploader;
