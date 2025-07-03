import { useState, useContext } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { Upload, FileAudio, Loader2, Play, BarChart3, Download } from "lucide-react";
import { api, ProcessAudioParams, WatermarkEmbedResponse } from "@/services/api";
import { AudioContext } from "@/contexts/AudioContext";
import { useWatermarkMethods } from "@/hooks/useWatermarkMethods";
import MethodComparisonChart from "@/components/comparison/MethodComparisonChart";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";

interface ResearchResult {
  method: string;
  snr_db: number;
  ber: number;
  detection_probability: number;
  processed_audio_url: string;
  timestamp: Date;
}

const Research = () => {
  // File state
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioName, setAudioName] = useState("");

  // Configuration state
  const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
  const [watermarkCount, setWatermarkCount] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);

  // Results state
  const [results, setResults] = useState<ResearchResult[]>([]);
  const [currentResult, setCurrentResult] = useState<ResearchResult | null>(null);

  // Hooks
  const { methods, isLoading: methodsLoading } = useWatermarkMethods();
  const { audioFile: contextAudioFile } = useContext(AudioContext);

  // Use context audio file if available
  const activeAudioFile = audioFile || contextAudioFile;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setAudioFile(files[0]);
      setAudioName(files[0].name);
      setResults([]);
      setCurrentResult(null);
    }
  };

  const handleMethodToggle = (method: string) => {
    setSelectedMethods(prev => 
      prev.includes(method) 
        ? prev.filter(m => m !== method)
        : [...prev, method]
    );
  };

  const generateRandomMessage = (length: number = 32) => {
    return Array.from({ length }, () => Math.round(Math.random())).join("");
  };

  const processMethod = async (method: string): Promise<ResearchResult> => {
    if (!activeAudioFile) throw new Error("No audio file selected");

    const response = await api.researchProcessAudio({
      audioFile: activeAudioFile,
      method: method,
      watermarkCount: watermarkCount
    });
    
    return {
      method: method.toUpperCase(),
      snr_db: response.results.snr_db,
      ber: response.results.ber,
      detection_probability: response.results.detection_probability,
      processed_audio_url: response.processed_audio_url,
      timestamp: new Date()
    };
  };

  const handleRunResearch = async () => {
    if (!activeAudioFile) {
      toast.error("Please select an audio file first");
      return;
    }

    if (selectedMethods.length === 0) {
      toast.error("Please select at least one method");
      return;
    }

    setIsProcessing(true);
    const newResults: ResearchResult[] = [];

    try {
      for (const method of selectedMethods) {
        toast.info(`Processing ${method.toUpperCase()}...`);
        
        try {
          const result = await processMethod(method);
          newResults.push(result);
          setCurrentResult(result);
          toast.success(`${method.toUpperCase()} completed successfully`);
        } catch (error: any) {
          console.error(`Error processing ${method}:`, error);
          toast.error(`Failed to process ${method.toUpperCase()}: ${error.message}`);
        }
      }

      setResults(prev => [...prev, ...newResults]);
      toast.success(`Research completed! Processed ${newResults.length} method(s)`);
    } catch (error: any) {
      console.error("Research error:", error);
      toast.error("Research failed: " + error.message);
    } finally {
      setIsProcessing(false);
      setCurrentResult(null);
    }
  };

  const handleClearResults = () => {
    setResults([]);
    setCurrentResult(null);
  };

  const availableMethods = Object.entries(methods).filter(([_, method]) => method.available);

  return (
    <div className="relative min-h-screen w-full overflow-x-hidden">
      <div className="grain-overlay" />
      <Header />
      
      <div className="max-w-6xl mx-auto pt-12 px-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Research Mode</h1>
          <p className="text-muted-foreground">
            Test and compare different audio watermarking methods
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Configuration */}
          <div className="lg:col-span-1 space-y-6">
            {/* File Upload */}
            <Card>
              <CardHeader>
                <CardTitle>1. Audio File</CardTitle>
              </CardHeader>
              <CardContent>
                <Label htmlFor="audio-upload-research" className="border-2 border-dashed rounded-lg p-6 text-center block cursor-pointer">
                  <input 
                    id="audio-upload-research" 
                    type="file" 
                    accept="audio/*" 
                    onChange={handleFileChange} 
                    className="hidden"
                    aria-label="Upload audio file for research"
                  />
                  {activeAudioFile ? (
                    <>
                      <FileAudio className="h-12 w-12 text-primary mx-auto mb-2" />
                      <p className="font-medium">{audioName}</p>
                      <p className="text-xs text-muted-foreground mt-1">Click to change file</p>
                    </>
                  ) : (
                    <>
                      <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                      <p className="font-medium">Click to upload audio</p>
                      <p className="text-xs text-muted-foreground mt-1">WAV, MP3, FLAC supported</p>
                    </>
                  )}
                </Label>
              </CardContent>
            </Card>

            {/* Method Selection */}
            <Card>
              <CardHeader>
                <CardTitle>2. Select Methods</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {methodsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin" />
                    <span className="ml-2">Loading methods...</span>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {availableMethods.map(([key, method]) => (
                      <div key={key} className="flex items-center space-x-3">
                        <Checkbox
                          id={`method-${key}`}
                          checked={selectedMethods.includes(key)}
                          onCheckedChange={() => handleMethodToggle(key)}
                        />
                        <label
                          htmlFor={`method-${key}`}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                        >
                          {method.name}
                        </label>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Watermark Count */}
            <Card>
              <CardHeader>
                <CardTitle>3. Watermark Count</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <Label>Number of Watermarks: {watermarkCount}</Label>
                  </div>
                  <Slider
                    min={1}
                    max={5}
                    step={1}
                    value={[watermarkCount]}
                    onValueChange={(value) => setWatermarkCount(value[0])}
                    className="py-4"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>1</span>
                    <span>3</span>
                    <span>5</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="space-y-3">
              <Button 
                onClick={handleRunResearch} 
                className="w-full" 
                disabled={isProcessing || !activeAudioFile || selectedMethods.length === 0}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Run Research
                  </>
                )}
              </Button>
              
              <Button 
                variant="outline" 
                onClick={handleClearResults} 
                className="w-full"
                disabled={results.length === 0}
              >
                Clear Results
              </Button>
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Current Processing Status */}
            {isProcessing && currentResult && (
              <Card>
                <CardHeader>
                  <CardTitle>Currently Processing</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">{currentResult.method}</p>
                      <p className="text-sm text-muted-foreground">
                        SNR: {currentResult.snr_db.toFixed(2)} dB | 
                        BER: {currentResult.ber.toFixed(4)} | 
                        Detection: {(currentResult.detection_probability * 100).toFixed(1)}%
                      </p>
                    </div>
                    <Loader2 className="h-6 w-6 animate-spin" />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Results Table */}
            {results.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Research Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Method</th>
                          <th className="text-left py-2">SNR (dB)</th>
                          <th className="text-left py-2">BER</th>
                          <th className="text-left py-2">Detection (%)</th>
                          <th className="text-left py-2">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.map((result, index) => (
                          <tr key={index} className="border-b">
                            <td className="py-2 font-medium">{result.method}</td>
                            <td className="py-2">{result.snr_db.toFixed(2)}</td>
                            <td className="py-2">{result.ber.toFixed(4)}</td>
                            <td className="py-2">{(result.detection_probability * 100).toFixed(1)}%</td>
                            <td className="py-2">
                              {result.processed_audio_url && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => window.open(result.processed_audio_url, '_blank')}
                                >
                                  <Download className="h-4 w-4" />
                                </Button>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Comparison Chart */}
            {results.length > 1 && (
              <Card>
                <CardHeader>
                  <CardTitle>Method Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <MethodComparisonChart results={results} />
                </CardContent>
              </Card>
            )}

            {/* Empty State */}
            {results.length === 0 && !isProcessing && (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">No Results Yet</h3>
                  <p className="text-muted-foreground text-center">
                    Upload an audio file, select methods, and run research to see results here.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default Research; 