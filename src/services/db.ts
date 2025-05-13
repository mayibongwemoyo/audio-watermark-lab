
import { toast } from "sonner";
import { api } from "./api";

const API_URL = "http://localhost:5000"; 

export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  created_at: string;
}

export interface AudioFile {
  id: number;
  filename: string;
  filepath: string;
  filehash: string;
  file_size: number;
  duration: number;
  sample_rate: number;
  upload_date: string;
  user_id: number | null;
}

export interface WatermarkEntry {
  id: number;
  action: "embed" | "detect";
  method: string;
  message: string;
  snr_db: number;
  detection_probability: number;
  ber: number;
  is_detected: boolean;
  purpose: string;
  watermark_count: number;
  metadata: Record<string, any>;
  created_at: string;
  user_id: number | null;
  audio_file_id: number | null;
  watermarked_file_id: number | null;
}

export interface ApiResponse<T> {
  status: "success" | "error";
  message?: string;
  [key: string]: any;
  data?: T;
}

export interface MethodStatistics {
  method: string;
  count: number;
  avg_snr: number;
  avg_detection: number;
  avg_ber: number;
}

// Helper function to handle API errors
async function handleApiResponse<T>(response: Response): Promise<ApiResponse<T>> {
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || `HTTP error ${response.status}`);
  }
  return await response.json();
}

export const dbApi = {
  /**
   * Initialize database with sample data
   */
  initializeDatabase: async (): Promise<boolean> => {
    try {
      const response = await fetch(`${API_URL}/init-db`);
      const data = await handleApiResponse(response);
      toast.success("Database initialized successfully");
      return true;
    } catch (error) {
      console.error("Failed to initialize database:", error);
      toast.error("Failed to initialize database");
      return false;
    }
  },

  /**
   * Get all users
   */
  getUsers: async (): Promise<User[]> => {
    try {
      const response = await fetch(`${API_URL}/api/users`);
      const data = await handleApiResponse<User[]>(response);
      return data.users || [];
    } catch (error) {
      console.error("Failed to fetch users:", error);
      toast.error("Failed to fetch users");
      return [];
    }
  },

  /**
   * Get all audio files
   */
  getAudioFiles: async (): Promise<AudioFile[]> => {
    try {
      const response = await fetch(`${API_URL}/api/audio_files`);
      const data = await handleApiResponse<AudioFile[]>(response);
      return data.audio_files || [];
    } catch (error) {
      console.error("Failed to fetch audio files:", error);
      toast.error("Failed to fetch audio files");
      return [];
    }
  },

  /**
   * Get watermark entries with optional filtering
   */
  getWatermarks: async (filters: Record<string, any> = {}): Promise<WatermarkEntry[]> => {
    try {
      // Build query string from filters
      const queryParams = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          queryParams.append(key, value.toString());
        }
      });

      const queryString = queryParams.toString();
      const url = `${API_URL}/api/watermarks${queryString ? `?${queryString}` : ''}`;
      
      const response = await fetch(url);
      const data = await handleApiResponse<WatermarkEntry[]>(response);
      return data.watermarks || [];
    } catch (error) {
      console.error("Failed to fetch watermarks:", error);
      toast.error("Failed to fetch watermark entries");
      return [];
    }
  },

  /**
   * Get method statistics
   */
  getMethodStatistics: async (): Promise<MethodStatistics[]> => {
    try {
      const response = await fetch(`${API_URL}/api/stats/methods`);
      const data = await handleApiResponse<MethodStatistics[]>(response);
      return data.statistics || [];
    } catch (error) {
      console.error("Failed to fetch method statistics:", error);
      toast.error("Failed to fetch method statistics");
      
      // Return mock data as fallback
      return await api.getMethodComparisons();
    }
  },
  
  /**
   * Create new watermark entry (manual method)
   */
  createWatermark: async (watermarkData: Partial<WatermarkEntry>): Promise<WatermarkEntry | null> => {
    try {
      const response = await fetch(`${API_URL}/api/watermarks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(watermarkData),
      });
      
      const data = await handleApiResponse<WatermarkEntry>(response);
      toast.success("Watermark entry created successfully");
      return data.watermark || null;
    } catch (error) {
      console.error("Failed to create watermark entry:", error);
      toast.error("Failed to create watermark entry");
      return null;
    }
  }
};
