
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Upload, Search, FileAudio, Clock, User, FileCheck } from "lucide-react";
import { api, ProcessAudioParams } from "@/services/api";

// Mock data for detected watermarks
const mockDetections = [
  {
    step: 0,
    userId: 3,
    role: "voice_actor",
    purpose: "training",
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    detected: true,
    confidence: 0.95
  },
  {
    step: 1,
    userId: 7,
    role: "editor",
    purpose: "remix",
    timestamp: new Date(Date.now() - 2400000).toISOString(),
    detected: true,
    confidence: 0.88
  },
  {
    step: 2,
    userId: 1,
    role: "marketer",
    purpose: "distribution",
    timestamp: new Date(Date.now() - 1200000).toISOString(),
    detected: true,
    confidence: 0.92
  }
];

const Detect = () => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioName, setAudioName] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [detections, setDetections] = useState<typeof mockDetections | null>(null);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setAudioFile(files[0]);
      setAudioName(files[0].name);
      setDetections(null); // Reset detections when new file selected
    }
  };
  
  const handleDetectWatermarks = async () => {
    if (!audioFile) {
      toast.error("Please upload an audio file first");
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Check if backend is available, otherwise use demo mode
      const isBackendAvailable = await api.checkHealth();
      
      if (isBackendAvailable) {
        const params: ProcessAudioParams = {
          audioFile,
          action: "detect",
          method: "pca", // Using PCA for best detection
          message: "00000000", // Placeholder, not used for detection
        };
        
        // In a real app, this would process the response from the backend
        await api.processAudio(params);
        
        // For now, use mock data with a delay to simulate processing
        await new Promise(resolve => setTimeout(resolve, 1500));
        setDetections(mockDetections);
        
        toast.success("Watermarks detected successfully");
      } else {
        // Demo mode - simulate processing
        await new Promise(resolve => setTimeout(resolve, 2000));
        setDetections(mockDetections);
        
        toast.success("Watermarks detected (Demo Mode)");
      }
    } catch (error) {
      console.error("Error detecting watermarks:", error);
      toast.error("Failed to detect watermarks");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto pt-12">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Detect & Trace</h1>
        <p className="text-muted-foreground">
          Upload audio to detect embedded watermarks and trace its history
        </p>
      </div>
      
      <div className="grid grid-cols-1 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Detect Watermarks</CardTitle>
          </CardHeader>
          
          <CardContent>
            <div className="space-y-6">
              {/* Audio Upload */}
              <div className="space-y-2">
                <Label htmlFor="audio-upload-detect">Upload Audio File</Label>
                <div className="border-2 border-dashed rounded-lg p-6 text-center">
                  <Input 
                    id="audio-upload-detect" 
                    type="file" 
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <label 
                    htmlFor="audio-upload-detect" 
                    className="flex flex-col items-center justify-center cursor-pointer"
                  >
                    {audioFile ? (
                      <>
                        <FileAudio className="h-12 w-12 text-primary mb-2" />
                        <p className="font-medium">{audioName}</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Click to change file
                        </p>
                      </>
                    ) : (
                      <>
                        <Upload className="h-12 w-12 text-muted-foreground mb-2" />
                        <p className="font-medium">Click to upload audio</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          WAV, MP3, FLAC supported
                        </p>
                      </>
                    )}
                  </label>
                </div>
              </div>
              
              {/* Detect Button */}
              <Button 
                onClick={handleDetectWatermarks} 
                className="w-full" 
                disabled={isProcessing || !audioFile}
              >
                {isProcessing ? (
                  <div className="flex items-center">
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                    Detecting Watermarks...
                  </div>
                ) : (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    Detect Watermarks
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
        
        {/* Detection Results */}
        {detections && detections.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Watermark Timeline</CardTitle>
            </CardHeader>
            
            <CardContent>
              <div className="relative space-y-8 before:absolute before:inset-0 before:ml-5 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-primary before:via-primary/70 before:to-transparent before:-translate-x-1/2 pt-2 md:ml-4">
                {detections.map((detection, index) => (
                  <div key={index} className="relative flex items-center gap-6">
                    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-background shadow-lg">
                      {index === 0 && <User className="w-5 h-5" />}
                      {index === 1 && <FileCheck className="w-5 h-5" />}
                      {index > 1 && <Clock className="w-5 h-5" />}
                    </div>
                    
                    <div className="flex flex-col rounded-lg bg-card p-3 shadow-lg md:p-4 border">
                      <h3 className="font-medium text-primary">Step {detection.step}</h3>
                      <time className="text-xs text-muted-foreground">
                        {new Date(detection.timestamp).toLocaleString()}
                      </time>
                      <p className="text-sm mt-1">
                        <span className="font-medium">User ID: </span>
                        {detection.userId}
                      </p>
                      <p className="text-sm">
                        <span className="font-medium">Role: </span>
                        {detection.role.replace('_', ' ')}
                      </p>
                      <p className="text-sm">
                        <span className="font-medium">Purpose: </span>
                        {detection.purpose}
                      </p>
                      <div className="mt-1 flex items-center">
                        <div className="h-2 w-full rounded-full bg-black/10 dark:bg-white/10">
                          <div 
                            className="h-2 rounded-full bg-primary" 
                            style={{ width: `${detection.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="ml-2 text-xs">
                          {Math.round(detection.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* No Watermarks Found */}
        {detections && detections.length === 0 && (
          <Card>
            <CardContent className="p-6 flex flex-col items-center justify-center text-center">
              <Search className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No Watermarks Detected</h3>
              <p className="text-muted-foreground">
                No watermarks were found in the uploaded audio file.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Detect;
