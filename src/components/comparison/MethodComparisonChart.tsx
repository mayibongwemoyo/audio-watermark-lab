import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MethodComparison, api } from "@/services/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from "recharts";

interface RadarData {
  subject: string;
  PCA: number;
  Blockchain: number;
  "Mobile Cloud": number;
}

const MethodComparisonChart = () => {
  const [comparisons, setComparisons] = useState<MethodComparison[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("snr");
  
  useEffect(() => {
    const fetchComparisons = async () => {
      try {
        setIsLoading(true);
        const data = await api.getMethodComparisons();
        setComparisons(data);
      } catch (error) {
        console.error("Error fetching method comparisons:", error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchComparisons();
  }, []);
  
  // Format data for different chart types
  const getBarData = (metric: keyof MethodComparison) => {
    return comparisons.map(comparison => ({
      name: comparison.name,
      value: comparison[metric],
    }));
  };
  
  // Format data specifically for radar chart
  const getRadarData = (): RadarData[] => {
    const metrics = ["SNR", "Detection", "BER", "Robustness"];
    return metrics.map(metric => {
      const lowerMetric = metric.toLowerCase();
      const pcaMethod = comparisons.find(c => c.name === "PCA");
      const blockchainMethod = comparisons.find(c => c.name === "Blockchain");
      const mcMethod = comparisons.find(c => c.name === "Mobile Cloud");
      
      // Initialize with default values of 0
      let value: RadarData = { 
        subject: metric, 
        PCA: 0, 
        Blockchain: 0, 
        "Mobile Cloud": 0 
      };
      
      if (pcaMethod && blockchainMethod && mcMethod) {
        switch(lowerMetric) {
          case "snr":
            value.PCA = Number(pcaMethod.snr || 0);
            value.Blockchain = Number(blockchainMethod.snr || 0);
            value["Mobile Cloud"] = Number(mcMethod.snr || 0);
            break;
          case "detection":
            value.PCA = Number(pcaMethod.detection_probability || 0) * 100;
            value.Blockchain = Number(blockchainMethod.detection_probability || 0) * 100;
            value["Mobile Cloud"] = Number(mcMethod.detection_probability || 0) * 100;
            break;
          case "ber":
            value.PCA = (1 - Number(pcaMethod.ber || 0)) * 100;
            value.Blockchain = (1 - Number(blockchainMethod.ber || 0)) * 100;
            value["Mobile Cloud"] = (1 - Number(mcMethod.ber || 0)) * 100;
            break;
          case "robustness":
            value.PCA = Number(pcaMethod.robustness || 0);
            value.Blockchain = Number(blockchainMethod.robustness || 0);
            value["Mobile Cloud"] = Number(mcMethod.robustness || 0);
            break;
        }
      }
      
      return value;
    });
  };
  
  const getSortedBarData = (metric: keyof MethodComparison) => {
    const barData = getBarData(metric);
    return [...barData].sort((a, b) => {
      // For BER, lower is better
      if (metric === "ber") {
        return Number(a.value) - Number(b.value);
      }
      // For others, higher is better
      return Number(b.value) - Number(a.value);
    });
  };
  
  const getMetricLabel = (metric: string): string => {
    switch(metric) {
      case "snr":
        return "Signal-to-Noise Ratio (dB)";
      case "ber":
        return "Bit Error Rate";
      case "detection_probability":
        return "Detection Probability";
      case "robustness":
        return "Robustness Score";
      default:
        return metric;
    }
  };
  
  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin w-8 h-8 border-4 border-black/20 dark:border-white/20 border-t-black dark:border-t-white rounded-full"></div>
        </div>
      </Card>
    );
  }
  
  return (
    <Card className="overflow-hidden">
      <div className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-5 mb-6">
            <TabsTrigger value="snr" className="rounded-full">SNR</TabsTrigger>
            <TabsTrigger value="ber" className="rounded-full">BER</TabsTrigger>
            <TabsTrigger value="detection" className="rounded-full">Detection</TabsTrigger>
            <TabsTrigger value="robustness" className="rounded-full">Robustness</TabsTrigger>
            <TabsTrigger value="radar" className="rounded-full">Overall</TabsTrigger>
          </TabsList>
          
          <TabsContent value="snr" className="min-h-[300px]">
            <h3 className="font-medium mb-4">Signal-to-Noise Ratio Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={getSortedBarData("snr")}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#55555522" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: "SNR (dB)", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="SNR (dB)" fill="#0066ff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </TabsContent>
          
          <TabsContent value="ber" className="min-h-[300px]">
            <h3 className="font-medium mb-4">Bit Error Rate Comparison (Lower is Better)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={getSortedBarData("ber")}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#55555522" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: "BER", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Bit Error Rate" fill="#ff6b6b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </TabsContent>
          
          <TabsContent value="detection" className="min-h-[300px]">
            <h3 className="font-medium mb-4">Detection Probability Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={getSortedBarData("detection_probability")}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#55555522" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: "Probability", angle: -90, position: "insideLeft" }} domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Detection Probability" fill="#5a91e3" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </TabsContent>
          
          <TabsContent value="robustness" className="min-h-[300px]">
            <h3 className="font-medium mb-4">Robustness Score Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={getSortedBarData("robustness")}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#55555522" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: "Robustness", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Robustness Score" fill="#5cb85c" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </TabsContent>
          
          <TabsContent value="radar" className="min-h-[300px]">
            <h3 className="font-medium mb-4">Overall Comparison (PCA vs Other Methods)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart outerRadius={90} width={730} height={250} data={getRadarData()}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar name="PCA" dataKey="PCA" stroke="#0066ff" fill="#0066ff" fillOpacity={0.6} />
                <Radar name="Blockchain" dataKey="Blockchain" stroke="#5cb85c" fill="#5cb85c" fillOpacity={0.6} />
                <Radar name="Mobile Cloud" dataKey="Mobile Cloud" stroke="#ff6b6b" fill="#ff6b6b" fillOpacity={0.6} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </TabsContent>
        </Tabs>
      </div>
    </Card>
  );
};

export default MethodComparisonChart;
