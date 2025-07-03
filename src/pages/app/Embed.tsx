import { useState, useContext } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Upload, FileAudio, Settings, Users } from "lucide-react";
import { api, ProcessAudioParams } from "@/services/api";
import { AudioPlayer } from "@/components/audio/AudioPlayer";
import { BatchProcessor } from "@/components/batch/BatchProcessor";
import { FileManager } from "@/components/files/FileManager";
import { AuthContext } from "@/contexts/AuthContext";

const Embed = () => {
  const { user: authenticatedUser } = useContext(AuthContext);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioName, setAudioName] = useState("");
  const [watermarkMessage, setWatermarkMessage] = useState("00000000000000000000000000000000");
  const [purpose, setPurpose] = useState("distribution");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedAudioUrl, setProcessedAudioUrl] = useState<string | null>(null);
  
  // PCA is the only method used in application mode
  const method = "pca_prime";
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setAudioFile(files[0]);
      setAudioName(files[0].name);
      setProcessedAudioUrl(null); // Reset processed audio when new file selected
    }
  };
  
  const handleEmbedWatermark = async () => {
    if (!audioFile) {
      toast.error("Please upload an audio file first");
      return;
    }
    
    if (watermarkMessage.length !== 32 || !/^[01]+$/.test(watermarkMessage)) {
      toast.error("Please enter a 32-bit binary watermark message (e.g., '1010...').");
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Check if backend is available, otherwise use demo mode
      const isBackendAvailable = await api.checkHealth();
      
      if (isBackendAvailable) {
        const params: ProcessAudioParams = {
          audioFile,
          action: "embed",
          method, // Always PCA in application mode
          message: watermarkMessage,
          purpose,
          watermarkCount: 4 // Ensure it's 4 for pca_prime
        };
        
        const result = await api.processAudio(params);
        
        // Type guard to check if result is WatermarkEmbedResponse
        if ('processed_audio_url' in result && result.processed_audio_url) {
          setProcessedAudioUrl(result.processed_audio_url);
          toast.success("Watermark embedded successfully using PCA Prime");
      } else {
          // Handle cases where result is missing or has an unexpected structure
          console.error("Unexpected result structure:", result);
          toast.error("Failed to embed watermark: Unexpected response");
      }
      } else {
        // Demo mode - simulate processing
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Create a mock processed audio URL (in real app, this would be the actual processed file)
        const mockUrl = URL.createObjectURL(audioFile);
        setProcessedAudioUrl(mockUrl);
        
        toast.success("Watermark embedded using PCA Prime (Demo Mode)");
      }
    } catch (error) {
      console.error("Error embedding watermark:", error);
      toast.error("Failed to embed watermark");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!processedAudioUrl) return;
    
    const link = document.createElement("a");
    link.href = processedAudioUrl;
    link.download = `watermarked_${audioName}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-5xl mx-auto pt-12">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Embed & Protect</h1>
        <p className="text-muted-foreground">
          Add watermarks to your audio files using PCA-based watermarking for optimal protection and tracking
        </p>
      </div>
      
      <Tabs defaultValue="single" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="single">Single File</TabsTrigger>
          <TabsTrigger value="batch">Batch Processing</TabsTrigger>
          <TabsTrigger value="manage">File Manager</TabsTrigger>
        </TabsList>
        
        <TabsContent value="single" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Single File Watermarking</CardTitle>
            </CardHeader>
            
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Upload and Settings */}
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
                  
                  {/* Watermark Settings */}
                  <div className="space-y-4">
                    {authenticatedUser && (
                      <div className="p-3 bg-muted rounded-md">
                        <p className="text-sm font-medium">Current User: {authenticatedUser.name || authenticatedUser.email}</p>
                        <p className="text-xs text-muted-foreground">Role: {authenticatedUser.role.replace('_', ' ')}</p>
                        <p className="text-xs text-muted-foreground">ID: {authenticatedUser.id}</p>
                      </div>
                    )}
                    <div className="space-y-2">
                      <Label htmlFor="watermark-message">Watermark Message</Label>
                      <Input
                        id="watermark-message"
                        value={watermarkMessage}
                        onChange={(e) => setWatermarkMessage(e.target.value)}
                        placeholder="Enter 32-bit binary message (e.g., 0000...0000)" // Updated placeholder
                        maxLength={32} // Ensure max length is 32
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Watermarking Method</Label>
                      <div className="flex items-center space-x-2 p-3 bg-muted rounded-md">
                        <div className="h-2 w-2 bg-primary rounded-full"></div>
                        <span className="font-medium">PCA Prime (Multi-Band NN)</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        PCA-based watermarking for optimal band selection and robustness
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="purpose-select">Purpose</Label>
                      <Select value={purpose} onValueChange={setPurpose}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="distribution">Distribution</SelectItem>
                          <SelectItem value="training">Training</SelectItem>
                          <SelectItem value="remix">Remix</SelectItem>
                          <SelectItem value="broadcast">Broadcast</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  {/* Process Button */}
                  <Button 
                    onClick={handleEmbedWatermark} 
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
                        <Settings className="h-4 w-4 mr-2" />
                        Embed Watermark
                      </>
                    )}
                  </Button>
                </div>
                
                {/* Preview and Results */}
                <div className="space-y-6">
                  {audioFile && (
                    <div>
                      <Label className="mb-2 block">Original Audio</Label>
                      <AudioPlayer 
                        audioUrl={URL.createObjectURL(audioFile)} 
                        fileName={`Original: ${audioName}`}
                      />
                    </div>
                  )}
                  
                  {processedAudioUrl && (
                    <div>
                      <Label className="mb-2 block">Watermarked Audio</Label>
                      <AudioPlayer 
                        audioUrl={processedAudioUrl} 
                        fileName={`Watermarked: ${audioName}`}
                        onDownload={handleDownload}
                      />
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="batch">
          <BatchProcessor />
        </TabsContent>
        
        <TabsContent value="manage">
          <FileManager />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Embed;
