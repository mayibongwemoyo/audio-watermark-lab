
import { toast } from "sonner";

const API_URL = "http://localhost:5000"; // Backend server URL

interface WatermarkMethod {
  name: string;
  description: string;
  available: boolean;
}

interface MethodsResponse {
  status: string;
  audioseal_available: boolean;
  methods: Record<string, WatermarkMethod>;
}

interface WatermarkEmbedResponse {
  status: string;
  action: string;
  method: string;
  message_embedded: string;
  snr_db: number;
  detection_probability: number;
  info: string;
  processed_audio_url: string;
}

interface WatermarkDetectResponse {
  status: string;
  action: string;
  method: string;
  message_checked: string;
  detection_probability: number;
  is_detected: boolean;
  ber: number;
  info: string;
}

export interface MethodComparison {
  name: string;
  snr: number;
  ber: number;
  detection_probability: number;
  robustness: number;
}

export const api = {
  /**
   * Get available watermarking methods from the backend
   */
  getMethods: async (): Promise<MethodsResponse> => {
    try {
      const response = await fetch(`${API_URL}/methods`);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error fetching methods:", error);
      toast.error("Failed to fetch available methods");
      return {
        status: "error",
        audioseal_available: false,
        methods: {
          placeholder: {
            name: "Placeholder",
            description: "Fallback implementation",
            available: true
          }
        }
      };
    }
  },
  
  /**
   * Check backend health status
   */
  checkHealth: async (): Promise<boolean> => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      return data.status === "healthy";
    } catch (error) {
      console.error("Backend connection error:", error);
      return false;
    }
  },
  
  /**
   * Process audio for watermarking or detection
   */
  processAudio: async (
    audioFile: File,
    action: "embed" | "detect",
    method: string,
    message: string
  ): Promise<WatermarkEmbedResponse | WatermarkDetectResponse> => {
    try {
      const formData = new FormData();
      formData.append("audio_file", audioFile);
      formData.append("action", action);
      formData.append("method", method);
      formData.append("message", message);
      
      const response = await fetch(`${API_URL}/process_audio`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error("Error processing audio:", error);
      toast.error("Failed to process audio");
      throw error;
    }
  },
  
  /**
   * Get comparison data between different watermark methods
   */
  getMethodComparisons: async (): Promise<MethodComparison[]> => {
    // This would be a real API call in production
    // For now we'll return mock data that matches your research
    return [
      {
        name: "PCA",
        snr: 48.5,
        ber: 0.08,
        detection_probability: 0.95,
        robustness: 92
      },
      {
        name: "SFA",
        snr: 29.2,
        ber: 0.42,
        detection_probability: 0.88,
        robustness: 74
      },
      {
        name: "SDA",
        snr: 31.3,
        ber: 0.38,
        detection_probability: 0.90,
        robustness: 78
      },
      {
        name: "PFB",
        snr: 55.5,
        ber: 0.48,
        detection_probability: 0.62,
        robustness: 65
      },
      {
        name: "Blockchain",
        snr: 45.0,
        ber: 0.12,
        detection_probability: 0.88,
        robustness: 85
      },
      {
        name: "Mobile Cloud",
        snr: 38.2,
        ber: 0.18,
        detection_probability: 0.82,
        robustness: 80
      }
    ];
  }
};
