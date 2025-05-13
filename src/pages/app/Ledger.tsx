
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Search, FileAudio, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

// Mock data for watermark ledger entries
const mockLedgerEntries = [
  {
    id: 1,
    userId: 3,
    userName: "Emma Jackson",
    role: "voice_actor",
    fileName: "voice_sample_01.wav",
    fileHash: "ae129f8d7c298a759f",
    purpose: "training",
    timestamp: new Date(Date.now() - 216000000).toISOString(),
    payload: "01100101"
  },
  {
    id: 2,
    userId: 5,
    userName: "Michael Chen",
    role: "producer",
    fileName: "track_mixdown_v2.wav",
    fileHash: "cb852e4f1d38b612a3",
    purpose: "internal",
    timestamp: new Date(Date.now() - 172800000).toISOString(),
    payload: "10101010"
  },
  {
    id: 3,
    userId: 7,
    userName: "Sophie Garcia",
    role: "editor",
    fileName: "final_mix_with_fx.wav",
    fileHash: "f2e901d45c781b324d",
    purpose: "remix",
    timestamp: new Date(Date.now() - 86400000).toISOString(),
    payload: "11000101"
  },
  {
    id: 4,
    userId: 2,
    userName: "David Kim",
    role: "producer",
    fileName: "podcast_intro.mp3",
    fileHash: "ba92c13e487f20d6e5",
    purpose: "distribution",
    timestamp: new Date(Date.now() - 43200000).toISOString(),
    payload: "00111001"
  },
  {
    id: 5,
    userId: 1,
    userName: "Alex Johnson",
    role: "marketer",
    fileName: "ad_campaign_audio.wav",
    fileHash: "d31fc9a8526b478209",
    purpose: "commercial",
    timestamp: new Date(Date.now() - 21600000).toISOString(),
    payload: "10010110"
  }
];

const Ledger = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRoles, setSelectedRoles] = useState<string[]>([]);
  
  const filteredEntries = mockLedgerEntries.filter(entry => {
    // Filter by search term
    const matchesSearch = 
      entry.fileName.toLowerCase().includes(searchTerm.toLowerCase()) || 
      entry.userName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.fileHash.toLowerCase().includes(searchTerm.toLowerCase());
      
    // Filter by selected roles
    const matchesRole = selectedRoles.length === 0 || selectedRoles.includes(entry.role);
    
    return matchesSearch && matchesRole;
  });
  
  const handleRoleToggle = (role: string) => {
    setSelectedRoles(prev => 
      prev.includes(role) 
        ? prev.filter(r => r !== role) 
        : [...prev, role]
    );
  };
  
  const roles = ["voice_actor", "producer", "editor", "marketer", "auditor"];

  return (
    <div className="max-w-5xl mx-auto pt-12">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Watermark Ledger</h1>
        <p className="text-muted-foreground">
          View the history of all watermarked audio files
        </p>
      </div>
      
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <Label htmlFor="search">Search</Label>
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="search"
                  placeholder="Search by filename, user or hash..."
                  className="pl-9"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>
            
            <div>
              <Label className="mb-2 block">Filter by Role</Label>
              <div className="flex flex-wrap gap-3">
                {roles.map((role) => (
                  <div key={role} className="flex items-center space-x-2">
                    <Checkbox 
                      id={`role-${role}`}
                      checked={selectedRoles.includes(role)}
                      onCheckedChange={() => handleRoleToggle(role)}
                    />
                    <label 
                      htmlFor={`role-${role}`}
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      {role.replace('_', ' ')}
                    </label>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Watermark History</CardTitle>
        </CardHeader>
        
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-3 px-4">Date</th>
                  <th className="text-left py-3 px-4">File</th>
                  <th className="text-left py-3 px-4">User</th>
                  <th className="text-left py-3 px-4">Role</th>
                  <th className="text-left py-3 px-4">Purpose</th>
                  <th className="text-left py-3 px-4">Payload</th>
                </tr>
              </thead>
              <tbody>
                {filteredEntries.map((entry) => (
                  <tr key={entry.id} className="hover:bg-black/5 dark:hover:bg-white/5">
                    <td className="py-3 px-4">
                      {new Date(entry.timestamp).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center">
                        <FileAudio className="h-4 w-4 mr-2 text-muted-foreground" />
                        <span>{entry.fileName}</span>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Hash: {entry.fileHash}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      {entry.userName}
                      <div className="text-xs text-muted-foreground">
                        ID: {entry.userId}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center rounded-full bg-blue-100 dark:bg-blue-900/30 px-2 py-1 text-xs">
                        {entry.role.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-1 text-xs">
                        {entry.purpose}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono">
                      {entry.payload}
                    </td>
                  </tr>
                ))}
                
                {filteredEntries.length === 0 && (
                  <tr>
                    <td colSpan={6} className="py-6 text-center text-muted-foreground">
                      No entries found matching your criteria
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 flex justify-end">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export Ledger
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Ledger;
