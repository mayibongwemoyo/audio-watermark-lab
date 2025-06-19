
import { useState } from "react";
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

const Embed = () => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioName, setAudioName] = useState("");
  const [watermarkMessage, setWatermarkMessage] = useState("00000000");
  const [method, setMethod] = useState("pca");
  const [purpose, setPurpose] = useState("distribution");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedAudioUrl, setProcessedAudioUrl] = useState<string | null>(null);
  
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
    
    if (!watermarkMessage.trim()) {
      toast.error("Please enter a watermark message");
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
          method,
          message: watermarkMessage,
          purpose
        };
        
        const result = await api.processAudio(params);
        
        if (result.processed_audio_url) {
          setProcessedAudioUrl(result.processed_audio_url);
          toast.success("Watermark embedded successfully");
        }
      } else {
        // Demo mode - simulate processing
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Create a mock processed audio URL (in real app, this would be the actual processed file)
        const mockUrl = URL.createObjectURL(audioFile);
        setProcessedAudioUrl(mockUrl);
        
        toast.success("Watermark embedded (Demo Mode)");
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
          Add watermarks to your audio files for protection and tracking
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
                    <div className="space-y-2">
                      <Label htmlFor="watermark-message">Watermark Message</Label>
                      <Input
                        id="watermark-message"
                        value={watermarkMessage}
                        onChange={(e) => setWatermarkMessage(e.target.value)}
                        placeholder="Enter binary message (e.g., 00000000)"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="method-select">Watermarking Method</Label>
                      <Select value={method} onValueChange={setMethod}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="pca">Principal Component Analysis (PCA)</SelectItem>
                          <SelectItem value="sfa">Sequential Fixed Alpha (SFA)</SelectItem>
                          <SelectItem value="sda">Sequential Decaying Alpha (SDA)</SelectItem>
                          <SelectItem value="pfb">Parallel Frequency Bands (PFB)</SelectItem>
                        </SelectContent>
                      </Select>
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
