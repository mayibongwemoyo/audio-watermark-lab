import { toast } from "sonner";

const API_URL = "http://localhost:5000"; // Backend server URL

// Add a simple retry mechanism
const fetchWithRetry = async (url: string, options: RequestInit, retries = 3): Promise<Response> => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, {
        ...options,
        mode: 'cors',
        credentials: 'omit',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });
      return response;
    } catch (error) {
      if (i === retries - 1) throw error;
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
  throw new Error('Max retries reached');
};

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

// export interface WatermarkEmbedResult {
//   step: number;
//   method: string;
//   message_embedded: string;
//   snr_db: number;
//   detection_probability: number;
//   ber: number;
//   info: string;
// }

// Define the shape of the CONSOLIDATED 'results' object returned by the backend's /process_audio (embed action)
export interface WatermarkEmbedResultsObject {
  snr_db: number;
  mse: number; // Add MSE as it's now returned by backend
  detection_probability: number;
  ber: number;
  is_detected: boolean;
  ber_per_band: number[]; // Add per-band BER array
  // You can add other fields from the backend's `results` object here if needed,
  // e.g., info?: string; message_embedded?: string;
}

export interface WatermarkEmbedResponse {
  status: string;
  action: string;
  method: string;
  // message_embedded: string;
  message_embedded_this_step: string
  // watermark_count: number;
  watermark_count_total?: number;
  processed_audio_url: string;
  results: WatermarkEmbedResultsObject;
  // results: WatermarkEmbedResult[];
  original_audio_file_id?: number;
  current_wm_idx?: number; 
}

export interface WatermarkDetectResponse {
  status: string;
  action: string;
  method: string;
  message_checked: string;
  detection_probability: number;
  is_detected: boolean;
  ber: number;
  ber_per_band?: number[]; // Add per-band BER for detection if backend sends it
  info: string;
  // New fields for application-mode detection
  detected_payload?: string;
  detected_per_band?: number[][];
}

export interface PreEmbedDetectResponse {
  status: 'success';
  detected_per_band: (string | number)[][];
  original_audio_file_id: number;
  current_audio_file_id: number;
  current_audio_file_path: string;
  next_wm_idx: number;
}

export interface MethodComparison {
  name: string;
  snr: number;
  ber: number;
  detection_probability: number;
  robustness: number;
}

export interface SingleWatermarkStepResult { // Renamed to avoid confusion
  step: number;
  method: string;
  message_embedded: string;
  snr_db: number;
  mse: number; // If historical steps also have MSE
  detection_probability: number;
  ber: number;
  is_detected: boolean;
  info?: string;
  // Add other fields relevant for a single step's data
}


export interface ProcessAudioParams {
  audioFile: File;
  action: "embed" | "detect";
  method: string;
  message: string;
  purpose?: string;
  watermarkCount?: number;
  pcaComponents?: number;
  // original_audio_file_id?: number; // Added for detection to link to original
  original_audio_file_id?: number; // The ID of the initial clean audio file
  current_wm_idx?: number; // The index (0-3) of the watermark being embedded in this step
  full_cumulative_message?: string; // The full 32-bit message for detector reference
}

// This interface is used by ResultsContext.tsx to store the overall results
// It should match the data structure that setResults expects in Dashboard.tsx
export interface WatermarkResultsContextType { // Renamed from WatermarkResults to avoid conflict
  snr_db: number;
  ber: number;
  detection_probability: number;
  processed_audio_url: string;
  method: string;
  mse?: number; // Optional MSE for display in context
  ber_per_band?: number[]; // Optional per-band BER for display in context
  step_results?: SingleWatermarkStepResult[]; // Optional: if you still want to manage historical step data
}

export interface WatermarkResults {
  snr_db: number;
  ber: number;
  detection_probability: number;
  processed_audio_url: string;
  method: string;
  // step_results: WatermarkEmbedResult[];
  results: WatermarkEmbedResultsObject;
}

export interface ResearchProcessParams {
  audioFile: File;
  method: string;
  watermarkCount: number;
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
      // Return a default/fallback object so the app doesn't crash
      return {
        status: "error",
        audioseal_available: false,
        methods: { /* your existing fallback methods */ }
      };
    }
  },
  
  /**
   * Check backend health status
   */
  checkHealth: async (): Promise<boolean> => {
    try {
      const response = await fetchWithRetry(`${API_URL}/health`, {
        method: 'GET',
      });
      
      if (!response.ok) {
        console.warn(`Backend health check failed with status: ${response.status}`);
        return false;
      }
      
      const data = await response.json();
      return data.status === "healthy";
    } catch (error) {
      console.error("Backend connection error:", error);
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        console.warn("Backend server is not running or not accessible");
      }
      return false;
    }
  },
  
  /**
   * Process audio for watermarking or detection
   */
  processAudio: async (params: ProcessAudioParams): Promise<WatermarkEmbedResponse | WatermarkDetectResponse> => {
    try {
      const formData = new FormData();
      formData.append("audio_file", params.audioFile);
      formData.append("action", params.action);
      formData.append("method", params.method);
      formData.append("message", params.message);
      
      // Append all other optional parameters from the new incremental flow
      if (params.purpose) formData.append("purpose", params.purpose);
      if (params.original_audio_file_id) formData.append("original_audio_file_id", params.original_audio_file_id.toString());
      if (params.current_wm_idx !== undefined) formData.append("current_wm_idx", params.current_wm_idx.toString());
      if (params.full_cumulative_message) formData.append("full_cumulative_message", params.full_cumulative_message);
      
      const response = await fetch(`${API_URL}/process_audio`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`HTTP error ${response.status}:`, errorText);
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      
      if (result.status === "error") {
        throw new Error(result.message || "Unknown server error");
      }
      
      return result;
    } catch (error) {
      console.error("Error processing audio:", error);
      
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        toast.error("Cannot connect to backend server. Please ensure the backend is running.");
      } else if (error instanceof Error) {
        toast.error(`Failed to process audio: ${error.message}`);
      } else {
        toast.error("Failed to process audio");
      }
      
      throw error;
    }
  },

  researchProcessAudio: async (params: ResearchProcessParams): Promise<WatermarkEmbedResponse> => {
    const formData = new FormData();
    formData.append("audio_file", params.audioFile);
    formData.append("method", params.method);
    formData.append("watermarkCount", String(params.watermarkCount));

    // Calls our new, dedicated research endpoint
    const response = await fetch(`${API_URL}/run_research_experiment`, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(60000) // 60-second timeout
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || "Research process failed");
    }

    return response.json();
  },

  /**
   * Analyzes an audio file before embedding to find existing watermarks.
   */
  preEmbedDetect: async (audioFile: File): Promise<PreEmbedDetectResponse> => {
    try {
        const formData = new FormData();
        formData.append("audio_file", audioFile);

        const response = await fetch(`${API_URL}/pre_embed_detect`, {
            method: "POST",
            body: formData,
            signal: AbortSignal.timeout(30000) // 30-second timeout
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`HTTP error ${response.status}:`, errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const result = await response.json();

        if (result.status === "error") {
            throw new Error(result.message || "Unknown server error during file analysis.");
        }

        return result;
    } catch (error) {
        console.error("Error during pre-embed analysis:", error);

        if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
            toast.error("Cannot connect to backend server for analysis.");
        } else if (error instanceof Error) {
            toast.error(`Analysis failed: ${error.message}`);
        } else {
            toast.error("An unknown error occurred during file analysis.");
        }
        
        throw error;
    }
  },
  
  /**
   * Get comparison data between different watermark methods
   */
  getMethodComparisons: async (): Promise<MethodComparison[]> => {
    // This is your existing mock data implementation
    return [
        { name: "PCA", snr: 48.5, ber: 0.08, detection_probability: 0.95, robustness: 92 },
        { name: "SFA", snr: 29.2, ber: 0.42, detection_probability: 0.88, robustness: 74 },
        { name: "SDA", snr: 31.3, ber: 0.38, detection_probability: 0.90, robustness: 78 },
        { name: "PFB", snr: 55.5, ber: 0.48, detection_probability: 0.62, robustness: 65 },
        { name: "Blockchain", snr: 45.0, ber: 0.12, detection_probability: 0.88, robustness: 85 },
        { name: "Mobile Cloud", snr: 38.2, ber: 0.18, detection_probability: 0.82, robustness: 80 }
    ];
  },

};
