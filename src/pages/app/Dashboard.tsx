
import { useState, useContext } from "react";
import { AuthContext } from "@/contexts/AuthContext";
import { AudioContext } from "@/contexts/AudioContext";
import { ResultsContext } from "@/contexts/ResultsContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { api, ProcessAudioParams, WatermarkEmbedResponse } from "@/services/api";
import { Upload, Wand2, Download, FileAudio } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// Purpose options for watermarking
const purposeOptions = [
  { id: "training", label: "Training Only", description: "Limited use for model training" },
  { id: "internal", label: "Internal Use", description: "Not for public distribution" },
  { id: "remix", label: "Remix Allowed", description: "Can be remixed with attribution" },
  { id: "commercial", label: "Commercial Use", description: "Can be used commercially" },
  { id: "distribution", label: "Distribution", description: "Can be publicly distributed" }
];

const Dashboard = () => {
  const { user } = useContext(AuthContext);
  const { audioFile, setAudioFile, audioName, setAudioName } = useContext(AudioContext);
  const { setResults } = useContext(ResultsContext);
  const [isProcessing, setIsProcessing] = useState(false);
  const [purpose, setPurpose] = useState("training");
  const [processedAudioUrl, setProcessedAudioUrl] = useState<string | null>(null);
  
  // Generate a simplified binary representation of user ID (3 bits)
  const getUserIdBits = () => {
    if (!user) return "000";
    const id = user.id % 8; // Ensure it fits in 3 bits (0-7)
    return id.toString(2).padStart(3, '0');
  };
  
  // Generate action ID based on role and purpose (2 bits)
  const getActionIdBits = () => {
    // Simple mapping of roles and purposes to 2-bit values
    const roleValue = user ? 
      ["voice_actor", "producer", "editor", "marketer", "auditor"].indexOf(user.role) % 2 : 0;
    
    const purposeValue = ["training", "internal", "remix", "commercial", "distribution"].indexOf(purpose) % 2;
    
    return (roleValue + purposeValue).toString(2).padStart(2, '0');
  };
  
  // Generate sequence bits (3 bits)
  const getSequenceBits = () => {
    // Simplified: just use current timestamp mod 8
    return (Math.floor(Date.now() / 1000) % 8).toString(2).padStart(3, '0');
  };
  
  const constructPayload = () => {
    // Combine all parts into an 8-bit payload
    const userBits = getUserIdBits();
    const actionBits = getActionIdBits();
    const sequenceBits = getSequenceBits();
    
    return userBits + actionBits + sequenceBits;
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setAudioFile(files[0]);
      setAudioName(files[0].name);
      setProcessedAudioUrl(null); // Reset processed audio when new file selected
    }
  };

  const handleApplyWatermark = async () => {
    if (!audioFile) {
      toast.error("Please upload an audio file first");
      return;
    }
    
    setIsProcessing(true);
    
    // Generate 8-bit payload
    const message = constructPayload();
    console.log("Generated payload:", message);
    
    try {
      const params: ProcessAudioParams = {
        audioFile,
        action: "embed",
        method: "pca", // Using PCA for best quality
        message,
        watermarkCount: 1,
        pcaComponents: 32
      };
      
      // Check if backend is available, otherwise use demo mode
      const isBackendAvailable = await api.checkHealth();
      
      if (isBackendAvailable) {
        const response = await api.processAudio(params) as WatermarkEmbedResponse;
        
        // Set results in context
        setResults({
          snr_db: response.results[response.results.length - 1].snr_db,
          ber: response.results[response.results.length - 1].ber,
          detection_probability: response.results[response.results.length - 1].detection_probability,
          processed_audio_url: response.processed_audio_url,
          method: response.method,
          step_results: response.results
        });
        
        setProcessedAudioUrl(response.processed_audio_url);
        
        // Log watermark action (in a real app, this would be saved to database)
        console.log("Watermark embedded:", {
          user: user?.id,
          role: user?.role,
          purpose,
          payload: message,
          timestamp: new Date().toISOString(),
          audioName
        });
        
        toast.success("Watermark successfully embedded");
      } else {
        // Demo mode - simulate processing
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        setResults({
          snr_db: 42.3,
          ber: 0.05,
          detection_probability: 0.94,
          processed_audio_url: URL.createObjectURL(audioFile), // Use original file in demo mode
          method: "pca",
          step_results: [{
            step: 1,
            method: "pca",
            message_embedded: message,
            snr_db: 42.3,
            detection_probability: 0.94,
            ber: 0.05,
            info: "Watermark embedded using PCA method (Demo Mode)"
          }]
        });
        
        setProcessedAudioUrl(URL.createObjectURL(audioFile));
        
        // Log demo watermark
        console.log("Demo watermark embedded:", {
          user: user?.id,
          role: user?.role,
          purpose,
          payload: message,
          timestamp: new Date().toISOString(),
          audioName
        });
        
        toast.success("Watermark embedded (Demo Mode)");
      }
    } catch (error) {
      console.error("Error embedding watermark:", error);
      toast.error("Failed to embed watermark");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto pt-12">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-muted-foreground">
          Embed watermarks into audio files with your identity
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="col-span-1 md:col-span-2">
          <CardHeader>
            <CardTitle>Embed Audio Watermark</CardTitle>
          </CardHeader>
          
          <CardContent>
            <div className="space-y-6">
              {/* Audio Upload */}
              <div className="space-y-2">
                <Label htmlFor="audio-upload">Upload Audio File</Label>
                <div className="border-2 border-dashed rounded-lg p-6 text-center">
                  <Input 
                    id="audio-upload" 
                    type="file" 
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <label 
                    htmlFor="audio-upload" 
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
              
              {/* Purpose Selection */}
              <div className="space-y-2">
                <Label htmlFor="purpose">Purpose</Label>
                <Select 
                  value={purpose} 
                  onValueChange={setPurpose}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select purpose" />
                  </SelectTrigger>
                  <SelectContent>
                    {purposeOptions.map((option) => (
                      <SelectItem key={option.id} value={option.id}>
                        {option.label} - {option.description}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  The purpose will be embedded in the watermark
                </p>
              </div>
              
              {/* Watermark Button */}
              <Button 
                onClick={handleApplyWatermark} 
                className="w-full" 
                disabled={isProcessing || !audioFile}
              >
                {isProcessing ? (
                  <div className="flex items-center">
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                    Embedding Watermark...
                  </div>
                ) : (
                  <>
                    <Wand2 className="h-4 w-4 mr-2" />
                    Embed Watermark
                  </>
                )}
              </Button>
              
              {/* Processed Audio Player */}
              {processedAudioUrl && (
                <div className="space-y-2">
                  <Label>Watermarked Audio</Label>
                  <div className="border rounded-md p-4">
                    <audio 
                      src={processedAudioUrl} 
                      controls 
                      className="w-full" 
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full mt-2"
                      onClick={() => {
                        const a = document.createElement("a");
                        a.href = processedAudioUrl;
                        a.download = `watermarked_${audioName || "audio.wav"}`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                      }}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download Watermarked Audio
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
        
        {/* Watermark Info Card */}
        <Card>
          <CardHeader>
            <CardTitle>Watermark Info</CardTitle>
          </CardHeader>
          
          <CardContent>
            <div className="space-y-4">
              <div className="space-y-1">
                <Label>User</Label>
                <div className="font-mono bg-black/5 dark:bg-white/5 p-2 rounded text-sm">
                  ID: {user?.id || "Unknown"}
                </div>
              </div>
              
              <div className="space-y-1">
                <Label>Role</Label>
                <div className="font-mono bg-black/5 dark:bg-white/5 p-2 rounded text-sm">
                  {user?.role?.replace('_', ' ') || "Unknown"}
                </div>
              </div>
              
              <div className="space-y-1">
                <Label>Purpose</Label>
                <div className="font-mono bg-black/5 dark:bg-white/5 p-2 rounded text-sm">
                  {purposeOptions.find(p => p.id === purpose)?.label || purpose}
                </div>
              </div>
              
              <div className="space-y-1">
                <Label>Timestamp</Label>
                <div className="font-mono bg-black/5 dark:bg-white/5 p-2 rounded text-sm">
                  {new Date().toLocaleString()}
                </div>
              </div>
              
              <div className="space-y-1">
                <Label>Payload</Label>
                <div className="font-mono bg-black/5 dark:bg-white/5 p-2 rounded text-sm overflow-auto">
                  {constructPayload()}
                </div>
              </div>
              
              <div className="space-y-1">
                <Label>Breakdown</Label>
                <div className="font-mono bg-black/5 dark:bg-white/5 p-2 rounded text-sm">
                  <div className="flex">
                    <span className="bg-blue-200 dark:bg-blue-800 px-1 mr-1">{getUserIdBits()}</span>
                    <span className="bg-green-200 dark:bg-green-800 px-1 mr-1">{getActionIdBits()}</span>
                    <span className="bg-purple-200 dark:bg-purple-800 px-1">{getSequenceBits()}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-1 text-xs mt-1">
                    <span>User ID</span>
                    <span>Action</span>
                    <span>Sequence</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
