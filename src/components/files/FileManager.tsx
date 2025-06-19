
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { 
  FileAudio, 
  Search, 
  Filter, 
  Download, 
  Trash2, 
  Eye,
  Share,
  Lock
} from "lucide-react";
import { AudioPlayer } from "@/components/audio/AudioPlayer";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface AudioFileItem {
  id: string;
  name: string;
  size: string;
  duration: string;
  format: string;
  watermarked: boolean;
  uploadDate: string;
  url: string;
}

// Mock data - in real app would come from database
const mockFiles: AudioFileItem[] = [
  {
    id: "1",
    name: "podcast_episode_001.wav",
    size: "24.5 MB",
    duration: "5:42",
    format: "WAV",
    watermarked: true,
    uploadDate: "2024-06-19",
    url: "/placeholder-audio.wav"
  },
  {
    id: "2", 
    name: "demo_track.mp3",
    size: "8.2 MB",
    duration: "3:28",
    format: "MP3",
    watermarked: false,
    uploadDate: "2024-06-18",
    url: "/placeholder-audio.mp3"
  }
];

export const FileManager = () => {
  const [files] = useState<AudioFileItem[]>(mockFiles);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterWatermarked, setFilterWatermarked] = useState<boolean | null>(null);

  const filteredFiles = files.filter(file => {
    const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterWatermarked === null || file.watermarked === filterWatermarked;
    return matchesSearch && matchesFilter;
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Audio Files</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Search and Filter */}
          <div className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search files..."
                className="pl-9"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="flex gap-2">
              <Button
                variant={filterWatermarked === null ? "default" : "outline"}
                size="sm"
                onClick={() => setFilterWatermarked(null)}
              >
                All
              </Button>
              <Button
                variant={filterWatermarked === true ? "default" : "outline"}
                size="sm"
                onClick={() => setFilterWatermarked(true)}
              >
                <Lock className="h-4 w-4 mr-1" />
                Watermarked
              </Button>
              <Button
                variant={filterWatermarked === false ? "default" : "outline"}
                size="sm"
                onClick={() => setFilterWatermarked(false)}
              >
                Original
              </Button>
            </div>
          </div>

          {/* File List */}
          <div className="space-y-2">
            {filteredFiles.map((file) => (
              <div key={file.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileAudio className="h-8 w-8 text-primary" />
                  <div>
                    <p className="font-medium">{file.name}</p>
                    <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                      <span>{file.size}</span>
                      <span>•</span>
                      <span>{file.duration}</span>
                      <span>•</span>
                      <Badge variant="outline" className="text-xs">
                        {file.format}
                      </Badge>
                      {file.watermarked && (
                        <Badge variant="default" className="text-xs">
                          <Lock className="h-3 w-3 mr-1" />
                          Protected
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button size="sm" variant="outline">
                        <Eye className="h-4 w-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-2xl">
                      <DialogHeader>
                        <DialogTitle>Preview: {file.name}</DialogTitle>
                      </DialogHeader>
                      <AudioPlayer 
                        audioUrl={file.url} 
                        fileName={file.name}
                        onDownload={() => console.log("Download", file.id)}
                      />
                    </DialogContent>
                  </Dialog>
                  
                  <Button size="sm" variant="outline">
                    <Share className="h-4 w-4" />
                  </Button>
                  
                  <Button size="sm" variant="outline">
                    <Download className="h-4 w-4" />
                  </Button>
                  
                  <Button size="sm" variant="outline">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>

          {filteredFiles.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              No files found matching your criteria
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
