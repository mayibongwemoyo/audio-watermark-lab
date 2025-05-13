
import { useState, useEffect, useCallback } from "react";
import { dbApi, WatermarkEntry, User, AudioFile, MethodStatistics } from "@/services/db";

export function useDatabaseStatus() {
  const [isInitialized, setIsInitialized] = useState<boolean | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const checkDatabaseStatus = useCallback(async () => {
    try {
      setIsLoading(true);
      // Try to fetch watermarks to see if DB is set up
      const entries = await dbApi.getWatermarks();
      setIsInitialized(entries.length > 0);
      setError(null);
    } catch (err) {
      setError("Failed to connect to database");
      setIsInitialized(false);
      console.error("Database status check failed:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  const initializeDatabase = useCallback(async () => {
    try {
      setIsLoading(true);
      const success = await dbApi.initializeDatabase();
      if (success) {
        setIsInitialized(true);
        setError(null);
      } else {
        setError("Failed to initialize database");
      }
    } catch (err) {
      setError("Failed to initialize database");
      console.error("Database initialization failed:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  useEffect(() => {
    checkDatabaseStatus();
  }, [checkDatabaseStatus]);
  
  return {
    isInitialized,
    isLoading,
    error,
    checkDatabaseStatus,
    initializeDatabase
  };
}

export function useWatermarkEntries(filters: Record<string, any> = {}) {
  const [entries, setEntries] = useState<WatermarkEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const fetchEntries = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await dbApi.getWatermarks(filters);
      setEntries(data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch watermark entries");
      console.error("Error fetching watermark entries:", err);
    } finally {
      setIsLoading(false);
    }
  }, [filters]);
  
  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);
  
  return {
    entries,
    isLoading,
    error,
    refetch: fetchEntries
  };
}

export function useMethodStatistics() {
  const [statistics, setStatistics] = useState<MethodStatistics[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const fetchStatistics = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await dbApi.getMethodStatistics();
      setStatistics(data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch method statistics");
      console.error("Error fetching method statistics:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  useEffect(() => {
    fetchStatistics();
  }, [fetchStatistics]);
  
  return {
    statistics,
    isLoading,
    error,
    refetch: fetchStatistics
  };
}

export function useUsers() {
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const fetchUsers = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await dbApi.getUsers();
      setUsers(data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch users");
      console.error("Error fetching users:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);
  
  return {
    users,
    isLoading,
    error,
    refetch: fetchUsers
  };
}

export function useAudioFiles() {
  const [audioFiles, setAudioFiles] = useState<AudioFile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const fetchAudioFiles = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await dbApi.getAudioFiles();
      setAudioFiles(data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch audio files");
      console.error("Error fetching audio files:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  useEffect(() => {
    fetchAudioFiles();
  }, [fetchAudioFiles]);
  
  return {
    audioFiles,
    isLoading,
    error,
    refetch: fetchAudioFiles
  };
}
