
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  Upload, 
  Play, 
  Pause, 
  CheckCircle, 
  XCircle, 
  Clock,
  FileAudio,
  Settings
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface BatchJob {
  id: string;
  fileName: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  method: string;
  message: string;
}

export const BatchProcessor = () => {
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchSettings, setBatchSettings] = useState({
    method: "pca",
    message: "00000000",
    purpose: "distribution"
  });

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newJobs: BatchJob[] = Array.from(files).map((file, index) => ({
      id: `job-${Date.now()}-${index}`,
      fileName: file.name,
      status: "pending",
      progress: 0,
      method: batchSettings.method,
      message: batchSettings.message
    }));

    setJobs(prev => [...prev, ...newJobs]);
  };

  const startBatchProcessing = async () => {
    setIsProcessing(true);
    
    for (let i = 0; i < jobs.length; i++) {
      if (jobs[i].status !== "pending") continue;
      
      // Update job to processing
      setJobs(prev => prev.map((job, idx) => 
        idx === i ? { ...job, status: "processing" as const } : job
      ));

      // Simulate processing with progress updates
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setJobs(prev => prev.map((job, idx) => 
          idx === i ? { ...job, progress } : job
        ));
      }

      // Mark as completed
      setJobs(prev => prev.map((job, idx) => 
        idx === i ? { ...job, status: "completed" as const, progress: 100 } : job
      ));
    }
    
    setIsProcessing(false);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pending": return <Clock className="h-4 w-4 text-yellow-500" />;
      case "processing": return <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />;
      case "completed": return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed": return <XCircle className="h-4 w-4 text-red-500" />;
      default: return null;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      pending: "secondary",
      processing: "default", 
      completed: "default", // Changed from "success" to "default"
      failed: "destructive"
    } as const;
    
    return (
      <Badge variant={variants[status as keyof typeof variants] || "secondary"}>
        {status}
      </Badge>
    );
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Batch Processing Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="batch-method">Watermark Method</Label>
              <Select value={batchSettings.method} onValueChange={(value) => 
                setBatchSettings(prev => ({ ...prev, method: value }))
              }>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pca">PCA</SelectItem>
                  <SelectItem value="sfa">SFA</SelectItem>
                  <SelectItem value="sda">SDA</SelectItem>
                  <SelectItem value="pfb">PFB</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="batch-message">Watermark Message</Label>
              <Input
                id="batch-message"
                value={batchSettings.message}
                onChange={(e) => setBatchSettings(prev => ({ ...prev, message: e.target.value }))}
                placeholder="00000000"
              />
            </div>
            
            <div>
              <Label htmlFor="batch-purpose">Purpose</Label>
              <Select value={batchSettings.purpose} onValueChange={(value) => 
                setBatchSettings(prev => ({ ...prev, purpose: value }))
              }>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="distribution">Distribution</SelectItem>
                  <SelectItem value="training">Training</SelectItem>
                  <SelectItem value="remix">Remix</SelectItem>
                  <SelectItem value="broadcast">Broadcast</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Upload Files for Batch Processing</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="border-2 border-dashed rounded-lg p-6 text-center">
              <Input 
                type="file" 
                multiple 
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
                id="batch-upload"
              />
              <label htmlFor="batch-upload" className="cursor-pointer">
                <Upload className="h-12 w-12 text-muted-foreground mb-2 mx-auto" />
                <p className="font-medium">Select multiple audio files</p>
                <p className="text-xs text-muted-foreground mt-1">
                  WAV, MP3, FLAC supported
                </p>
              </label>
            </div>

            {jobs.length > 0 && (
              <div className="flex justify-between items-center">
                <p className="text-sm text-muted-foreground">
                  {jobs.length} file(s) queued
                </p>
                <Button 
                  onClick={startBatchProcessing} 
                  disabled={isProcessing || jobs.every(job => job.status !== "pending")}
                >
                  {isProcessing ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                  {isProcessing ? "Processing..." : "Start Batch"}
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {jobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Processing Queue</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {jobs.map((job) => (
                <div key={job.id} className="flex items-center space-x-4 p-3 border rounded-lg">
                  <FileAudio className="h-6 w-6 text-primary" />
                  
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <p className="font-medium text-sm">{job.fileName}</p>
                      {getStatusBadge(job.status)}
                    </div>
                    
                    <div className="flex items-center space-x-2 text-xs text-muted-foreground mb-2">
                      <span>{job.method.toUpperCase()}</span>
                      <span>•</span>
                      <span>{job.message}</span>
                    </div>
                    
                    {(job.status === "processing" || job.status === "completed") && (
                      <Progress value={job.progress} className="h-2" />
                    )}
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(job.status)}
                    {job.status === "processing" && (
                      <span className="text-xs">{job.progress}%</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
