import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { BarChart, Waves, FileDown, Lock, BarChart2, Check, Info } from "lucide-react";
import { BarChart as ReBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const AnalysisSection = () => {
  const [activeTab, setActiveTab] = useState("metrics");
  
  // Sample data for visualization
  const berData = [
    { name: 'Original', watermark1: 0, watermark2: 0, watermark3: 0 },
    { name: 'Noise', watermark1: 0.05, watermark2: 0.08, watermark3: 0.12 },
    { name: 'MP3', watermark1: 0.12, watermark2: 0.18, watermark3: 0.25 },
    { name: 'Resample', watermark1: 0.08, watermark2: 0.15, watermark3: 0.22 },
  ];
  
  const comparisonData = [
    { name: 'PCA', BER: 0.08, SNR: 35 },
    { name: 'LSB', BER: 0.22, SNR: 30 },
    { name: 'SS', BER: 0.15, SNR: 32 },
  ];
  
  return (
    <section id="analysis" className="py-24 px-6 md:px-10">
      <div className="max-w-4xl mx-auto">
        <div className="flex flex-col gap-3 mb-12 animate-fade-in">
          <div className="flex items-center gap-2">
            <div className="h-8 w-1 bg-black dark:bg-white rounded-full"></div>
            <h2 className="text-xl md:text-2xl font-semibold">Analysis & Results</h2>
          </div>
          <p className="text-black/70 dark:text-white/70 ml-3 pl-3">
            Explore the performance metrics and comparative analysis of your watermarking
          </p>
        </div>
        
        <Card className="rounded-xl overflow-hidden animate-scale-in mb-8">
          <div className="p-6">
            <Tabs defaultValue="metrics" onValueChange={setActiveTab}>
              <TabsList className="grid grid-cols-3 mb-6">
                <TabsTrigger value="metrics" className="rounded-full">
                  <BarChart size={16} className="mr-2" />
                  Metrics
                </TabsTrigger>
                <TabsTrigger value="comparison" className="rounded-full">
                  <BarChart2 size={16} className="mr-2" />
                  Comparison
                </TabsTrigger>
                <TabsTrigger value="info" className="rounded-full">
                  <Info size={16} className="mr-2" />
                  Research Info
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="metrics" className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">BER (Bit Error Rate)</span>
                      <span className="text-sm font-medium">8.3%</span>
                    </div>
                    <Progress value={8.3} className="h-2" />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">SNR (Signal-to-Noise)</span>
                      <span className="text-sm font-medium">35.7 dB</span>
                    </div>
                    <Progress value={85} className="h-2" />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">PESQ</span>
                      <span className="text-sm font-medium">4.2</span>
                    </div>
                    <Progress value={84} className="h-2" />
                  </div>
                </div>
                
                <div className="bg-black/5 dark:bg-white/5 rounded-lg p-4">
                  <h3 className="font-medium mb-3">Watermark Robustness</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <ReBarChart
                        data={berData}
                        margin={{
                          top: 5,
                          right: 30,
                          left: 20,
                          bottom: 5,
                        }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#55555522" />
                        <XAxis dataKey="name" />
                        <YAxis label={{ value: 'BER', angle: -90, position: 'insideLeft' }} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.8)', 
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                            borderColor: 'rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Legend />
                        <Bar dataKey="watermark1" name="Watermark 1" fill="#0066ff" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="watermark2" name="Watermark 2" fill="#00aaff" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="watermark3" name="Watermark 3" fill="#00ddff" radius={[4, 4, 0, 0]} />
                      </ReBarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Check size={20} className="text-green-500" />
                    <span className="font-medium">All watermarks successfully detected</span>
                  </div>
                  <Button 
                    variant="outline"
                    size="sm"
                    className="rounded-full"
                  >
                    <FileDown size={16} className="mr-2" />
                    Export Results
                  </Button>
                </div>
              </TabsContent>
              
              <TabsContent value="comparison" className="space-y-6">
                <div className="bg-black/5 dark:bg-white/5 rounded-lg p-4">
                  <h3 className="font-medium mb-3">H1: PCA vs Traditional Methods</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <ReBarChart
                        data={comparisonData}
                        margin={{
                          top: 5,
                          right: 30,
                          left: 20,
                          bottom: 5,
                        }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#55555522" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.8)', 
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                            borderColor: 'rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Legend />
                        <Bar dataKey="BER" name="Bit Error Rate" fill="#ff6b6b" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="SNR" name="Signal-to-Noise Ratio" fill="#5a91e3" radius={[4, 4, 0, 0]} />
                      </ReBarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-black/5 dark:bg-white/5 p-4 rounded-lg flex flex-col items-center text-center">
                    <Waves size={30} className="mb-2 text-blue-500" />
                    <h3 className="font-medium mb-1">PCA Method</h3>
                    <p className="text-sm text-black/60 dark:text-white/60 mb-2">
                      Dynamic band selection
                    </p>
                    <div className="mt-auto">
                      <span className="text-xs px-2 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-100 rounded-full">
                        Recommended
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-black/5 dark:bg-white/5 p-4 rounded-lg flex flex-col items-center text-center">
                    <BarChart size={30} className="mb-2 text-orange-500" />
                    <h3 className="font-medium mb-1">LSB Method</h3>
                    <p className="text-sm text-black/60 dark:text-white/60 mb-2">
                      Basic bit replacement
                    </p>
                    <div className="mt-auto">
                      <span className="text-xs px-2 py-1 bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-100 rounded-full">
                        Limited robustness
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-black/5 dark:bg-white/5 p-4 rounded-lg flex flex-col items-center text-center">
                    <Lock size={30} className="mb-2 text-purple-500" />
                    <h3 className="font-medium mb-1">Blockchain</h3>
                    <p className="text-sm text-black/60 dark:text-white/60 mb-2">
                      External verification
                    </p>
                    <div className="mt-auto">
                      <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 rounded-full">
                        High security
                      </span>
                    </div>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="info" className="space-y-6">
                <div className="bg-black/5 dark:bg-white/5 rounded-lg p-6">
                  <h3 className="font-semibold mb-4">Research Overview</h3>
                  <p className="text-black/80 dark:text-white/80 mb-4">
                    This project explores a novel approach to audio watermarking using Principal Component Analysis 
                    (PCA) to identify optimal embedding locations in the frequency domain. We demonstrate superior 
                    robustness against common attacks compared to traditional methods.
                  </p>
                  
                  <h4 className="font-medium mt-6 mb-2">Key Hypotheses</h4>
                  <ul className="list-disc list-inside space-y-2 text-black/80 dark:text-white/80">
                    <li>H1: PCA-based watermarking provides better robustness than traditional methods</li>
                    <li>H2: Multi-layer embedding maintains detection accuracy even under severe attacks</li>
                  </ul>
                  
                  <div className="mt-6 pt-6 border-t border-black/10 dark:border-white/10">
                    <div className="flex items-center gap-2">
                      <img 
                        src="https://seeklogo.com/images/U/university-of-zimbabwe-logo-26612F49B4-seeklogo.com.png" 
                        alt="University of Zimbabwe" 
                        className="h-12 w-12 object-contain" 
                      />
                      <div>
                        <p className="text-sm font-medium">University of Zimbabwe</p>
                        <p className="text-xs text-black/60 dark:text-white/60">Faculty of Engineering</p>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </Card>
        
        <div className="flex justify-center mb-12">
          <Button
            variant="outline"
            className="rounded-full px-8"
            onClick={() => {
              document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            Learn More About the Research
          </Button>
        </div>
      </div>
    </section>
  );
};

export default AnalysisSection;
