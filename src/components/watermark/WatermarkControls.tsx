
import { useState } from "react";
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
  SquareDashedBottom
} from "lucide-react";
import { toast } from "sonner";

const WatermarkControls = () => {
  const [watermarkCount, setWatermarkCount] = useState(1);
  const [messageBits, setMessageBits] = useState(32);
  const [pcaEnabled, setPcaEnabled] = useState(true);
  const [pcaComponents, setPcaComponents] = useState(32);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const handleApplyWatermark = () => {
    setIsProcessing(true);
    
    // Simulate processing delay
    setTimeout(() => {
      setIsProcessing(false);
      toast.success("Watermark applied successfully");
      document.getElementById('analysis')?.scrollIntoView({ behavior: 'smooth' });
    }, 2000);
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
                    
                    <div className="pt-2 pb-2">
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <Label htmlFor="pca-toggle">Enable PCA</Label>
                          <p className="text-xs text-muted-foreground">Use Principal Component Analysis for embedding</p>
                        </div>
                        <Switch 
                          id="pca-toggle" 
                          checked={pcaEnabled}
                          onCheckedChange={setPcaEnabled}
                        />
                      </div>
                    </div>
                    
                    {pcaEnabled && (
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
                          disabled={!pcaEnabled}
                          className="py-4"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>1</span>
                          <span>32</span>
                          <span>64</span>
                        </div>
                      </div>
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
                  <span className="text-sm text-black/70 dark:text-white/70">Watermarks</span>
                  <span className="font-medium">{watermarkCount}</span>
                </div>
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">Message Size</span>
                  <span className="font-medium">{messageBits} bits</span>
                </div>
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">PCA</span>
                  <span className="font-medium">{pcaEnabled ? 'Enabled' : 'Disabled'}</span>
                </div>
                
                {pcaEnabled && (
                  <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                    <span className="text-sm text-black/70 dark:text-white/70">Components</span>
                    <span className="font-medium">{pcaComponents}</span>
                  </div>
                )}
                
                <div className="flex justify-between items-center py-2 border-b border-black/10 dark:border-white/10">
                  <span className="text-sm text-black/70 dark:text-white/70">Total Capacity</span>
                  <span className="font-medium">{watermarkCount * messageBits} bits</span>
                </div>
              </div>
            </div>
            
            <div className="px-6 pb-6">
              <Button 
                variant="outline"
                className="w-full rounded-full"
                disabled
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
