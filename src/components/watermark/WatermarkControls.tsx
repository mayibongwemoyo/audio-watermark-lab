
import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Layers, 
  Settings, 
  BarChart2, 
  Shield, 
  Play,
  FileDown,
  SquareDashedBottom,
  AlertTriangle
} from "lucide-react";
import { toast } from "sonner";
import { useWatermarkMethods } from "@/hooks/useWatermarkMethods";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { api } from "@/services/api";

const WatermarkControls = () => {
  const [watermarkCount, setWatermarkCount] = useState(1);
  const [messageBits, setMessageBits] = useState(32);
  const [pcaEnabled, setPcaEnabled] = useState(true);
  const [pcaComponents, setPcaComponents] = useState(32);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState("sfa");
  const [backendConnected, setBackendConnected] = useState(false);
  
  const { methods, audioSealAvailable, isLoading: methodsLoading } = useWatermarkMethods();
  
  // Check backend connection
  useEffect(() => {
    const checkBackendConnection = async () => {
      const isConnected = await api.checkHealth();
      setBackendConnected(isConnected);
      
      if (!isConnected) {
        toast.warning("Backend server not detected. Running in demo mode.", {
          duration: 5000,
        });
      }
    };
    
    checkBackendConnection();
  }, []);
  
  const handleApplyWatermark = () => {
    setIsProcessing(true);
    
    // Generate a binary message string based on messageBits length
    const generateRandomBinaryMessage = (length: number) => {
      return Array.from({ length }, () => Math.round(Math.random())).join("");
    };
    
    const message = generateRandomBinaryMessage(messageBits);
    
    // If backend is connected, process with real API
    if (backendConnected) {
      // This would normally use the real audio file from AudioUploader component
      // For demo purposes we'll just simulate the response
      setTimeout(() => {
        setIsProcessing(false);
        toast.success(`Watermark applied using ${selectedMethod.toUpperCase()} method`);
        document.getElementById('analysis')?.scrollIntoView({ behavior: 'smooth' });
      }, 2000);
    } else {
      // Simulate processing delay in demo mode
      setTimeout(() => {
        setIsProcessing(false);
        toast.success("Watermark applied successfully (Demo Mode)");
        document.getElementById('analysis')?.scrollIntoView({ behavior: 'smooth' });
      }, 2000);
    }
  };
  
  return (
    <section id="watermark" className="py-24 px-6 md:px-10 bg-black/5 dark:bg-white/5">
      <div className="max-w-4xl mx-auto">
        <div className="flex flex-col gap-3 mb-12 animate-fade-in">
          <div className="flex items-center gap-2">
            <div className="h-8 w-1 bg-black dark:bg-white rounded-full"></div>
            <h2 className="text-xl md:text-2xl font-semibold">Watermark Configuration</h2>
          </div>
          <p className="text-black/70 dark:text-white/70 ml-3 pl-3">
            Configure the parameters for your audio watermarking process
          </p>
          
          {!backendConnected && (
            <div className="ml-3 pl-3 mt-2 flex items-center gap-2 text-yellow-600 dark:text-yellow-400">
              <AlertTriangle size={16} />
              <span className="text-sm">
                Running in demo mode. Backend server not connected.
              </span>
            </div>
          )}
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="col-span-1 md:col-span-2 animate-slide-up">
            <div className="p-6">
              <Tabs defaultValue="watermark">
                <TabsList className="grid grid-cols-3 mb-6">
                  <TabsTrigger value="watermark" className="rounded-full">
                    <Layers size={16} className="mr-2" />
                    Watermark
                  </TabsTrigger>
                  <TabsTrigger value="attacks" className="rounded-full">
                    <Shield size={16} className="mr-2" />
                    Attacks
                  </TabsTrigger>
                  <TabsTrigger value="analysis" className="rounded-full">
                    <BarChart2 size={16} className="mr-2" />
                    Analysis
                  </TabsTrigger>
                </TabsList>
                
                <TabsContent value="watermark" className="space-y-6">
                  <div className="space-y-4">
                    <div className="pt-2 pb-2">
                      <Label htmlFor="method" className="block mb-2">Watermarking Method</Label>
                      <Select 
                        value={selectedMethod} 
                        onValueChange={setSelectedMethod}
                        disabled={methodsLoading}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select method" />
                        </SelectTrigger>
                        <SelectContent>
                          {methodsLoading ? (
                            <SelectItem value="loading">Loading methods...</SelectItem>
                          ) : (
                            Object.entries(methods).map(([key, method]) => (
                              <SelectItem key={key} value={key}>
                                {method.name}
                              </SelectItem>
                            ))
                          )}
                          {!backendConnected && (
                            <>
                              <SelectItem value="sfa">Sequential Fixed Alpha (SFA)</SelectItem>
                              <SelectItem value="sda">Sequential Decaying Alpha (SDA)</SelectItem>
                              <SelectItem value="pfb">Parallel Frequency Bands (PFB)</SelectItem>
                              <SelectItem value="pca">Principal Component Analysis (PCA)</SelectItem>
                            </>
                          )}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground mt-1">
                        {selectedMethod === "pca" 
                          ? "PCA-based watermarking for optimal band selection"
                          : methods[selectedMethod]?.description || "Select a watermarking method"}
                      </p>
                    </div>
                    
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <Label htmlFor="watermark-count">Number of Watermarks: {watermarkCount}</Label>
                      </div>
                      <Slider
                        id="watermark-count"
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
                    
                    <div className="pt-4">
                      <div className="flex justify-between items-center mb-2">
                        <Label htmlFor="message-bits">Message Bits per Watermark: {messageBits}</Label>
                      </div>
                      <Slider
                        id="message-bits"
                        min={16}
                        max={64}
                        step={16}
                        value={[messageBits]}
                        onValueChange={(value) => setMessageBits(value[0])}
                        className="py-4"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>16</span>
                        <span>32</span>
                        <span>64</span>
                      </div>
                    </div>
                    
                    {(selectedMethod === "pca" || pcaEnabled) && (
                      <>
                        <div className="pt-2 pb-2">
                          <div className="flex items-center justify-between">
                            <div className="space-y-1">
                              <Label htmlFor="pca-toggle">Enable PCA</Label>
                              <p className="text-xs text-muted-foreground">Use Principal Component Analysis for embedding</p>
                            </div>
                            <Switch 
                              id="pca-toggle" 
                              checked={pcaEnabled || selectedMethod === "pca"}
                              onCheckedChange={setPcaEnabled}
                              disabled={selectedMethod === "pca"}
                            />
                          </div>
                        </div>
                        
                        <div className="pt-2">
                          <div className="flex justify-between items-center mb-2">
                            <Label htmlFor="pca-components">PCA Components: {pcaComponents}</Label>
                          </div>
                          <Slider
                            id="pca-components"
                            min={1}
                            max={64}
                            step={1}
                            value={[pcaComponents]}
                            onValueChange={(value) => setPcaComponents(value[0])}
                            disabled={!pcaEnabled && selectedMethod !== "pca"}
                            className="py-4"
                          />
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>1</span>
                            <span>32</span>
                            <span>64</span>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </TabsContent>
                
                <TabsContent value="attacks" className="min-h-[300px] flex flex-col items-center justify-center text-center p-8">
                  <SquareDashedBottom size={48} className="mb-4 text-black/40 dark:text-white/40" />
                  <h3 className="text-lg font-medium mb-2">Attack Simulation</h3>
                  <p className="text-black/60 dark:text-white/60 mb-4">
                    Apply watermarks first to enable attack simulation options
                  </p>
                </TabsContent>
                
                <TabsContent value="analysis" className="min-h-[300px] flex flex-col items-center justify-center text-center p-8">
                  <BarChart2 size={48} className="mb-4 text-black/40 dark:text-white/40" />
                  <h3 className="text-lg font-medium mb-2">Watermark Analysis</h3>
                  <p className="text-black/60 dark:text-white/60 mb-4">
                    Apply watermarks first to view analysis results
                  </p>
                </TabsContent>
              </Tabs>
            </div>
            
            <div className="px-6 pb-6 flex gap-4">
              <Button 
                variant="outline"
                className="rounded-full flex-1"
                onClick={() => {
                  toast.info("Configuration reset");
                  setWatermarkCount(1);
                  setMessageBits(32);
                  setPcaEnabled(true);
                  setPcaComponents(32);
                  setSelectedMethod("sfa");
                }}
              >
                <Settings size={16} className="mr-2" />
                Reset
              </Button>
              <Button 
                className="rounded-full flex-1 bg-black hover:bg-black/90 text-white dark:bg-white dark:text-black dark:hover:bg-white/90"
                onClick={handleApplyWatermark}
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <div className="flex items-center">
                    <div className="animate-spin w-4 h-4 border-2 border-white dark:border-black border-t-transparent dark:border-t-transparent rounded-full mr-2"></div>
                    Processing...
                  </div>
                ) : (
                  <>
                    <Play size={16} className="mr-2" />
                    Apply Watermark
                  </>
                )}
              </Button>
            </div>
          </Card>
          
          <Card className="animate-slide-up animation-delay-100">
            <div className="p-6">
              <h3 className="text-lg font-medium mb-4">Summary</h3>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">Method</span>
                  <span className="font-medium">{selectedMethod.toUpperCase()}</span>
                </div>
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">Watermarks</span>
                  <span className="font-medium">{watermarkCount}</span>
                </div>
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">Message Size</span>
                  <span className="font-medium">{messageBits} bits</span>
                </div>
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">PCA</span>
                  <span className="font-medium">{(pcaEnabled || selectedMethod === "pca") ? 'Enabled' : 'Disabled'}</span>
                </div>
                
                {(pcaEnabled || selectedMethod === "pca") && (
                  <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                    <span className="text-sm text-black/70 dark:text-white/70">Components</span>
                    <span className="font-medium">{pcaComponents}</span>
                  </div>
                )}
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">Total Capacity</span>
                  <span className="font-medium">{watermarkCount * messageBits} bits</span>
                </div>
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">AudioSeal</span>
                  {methodsLoading ? (
                    <span className="text-sm">Checking...</span>
                  ) : (
                    <span className="font-medium">{audioSealAvailable ? 'Available' : 'Not detected'}</span>
                  )}
                </div>
              </div>
            </div>
            
            <div className="px-6 pb-6">
              <Button 
                variant="outline"
                className="w-full rounded-full"
                disabled={!backendConnected}
              >
                <FileDown size={16} className="mr-2" />
                Download Config
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default WatermarkControls;
