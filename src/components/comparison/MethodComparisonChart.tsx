import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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

interface ResearchResult {
  method: string;
  snr_db: number;
  ber: number;
  detection_probability: number;
  processed_audio_url: string;
  timestamp: Date;
}

interface MethodComparisonChartProps {
  results: ResearchResult[];
}

const MethodComparisonChart = ({ results }: MethodComparisonChartProps) => {
  if (results.length === 0) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          No results to display
        </div>
      </Card>
    );
  }

  // Format data for different chart types
  const getBarData = (metric: keyof Pick<ResearchResult, 'snr_db' | 'ber' | 'detection_probability'>) => {
    return results.map(result => ({
      name: result.method,
      value: result[metric],
    }));
  };

  const getSortedBarData = (metric: keyof Pick<ResearchResult, 'snr_db' | 'ber' | 'detection_probability'>) => {
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
      case "snr_db":
        return "Signal-to-Noise Ratio (dB)";
      case "ber":
        return "Bit Error Rate";
      case "detection_probability":
        return "Detection Probability";
      default:
        return metric;
    }
  };

  return (
    <Card className="overflow-hidden">
      <div className="p-6">
        <Tabs defaultValue="snr">
          <TabsList className="grid grid-cols-3 mb-6">
            <TabsTrigger value="snr" className="rounded-full">SNR</TabsTrigger>
            <TabsTrigger value="ber" className="rounded-full">BER</TabsTrigger>
            <TabsTrigger value="detection" className="rounded-full">Detection</TabsTrigger>
          </TabsList>
          
          <TabsContent value="snr" className="min-h-[300px]">
            <h3 className="font-medium mb-4">Signal-to-Noise Ratio Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={getSortedBarData("snr_db")}
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
        </Tabs>
      </div>
    </Card>
  );
};

export default MethodComparisonChart;
