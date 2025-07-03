import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Search, FileAudio, Download, Database, RefreshCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { dbApi, WatermarkEntry } from "@/services/db";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";

const Ledger = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRoles, setSelectedRoles] = useState<string[]>([]);
  const [entries, setEntries] = useState<WatermarkEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isInitializing, setIsInitializing] = useState(false);
  const [dbInitialized, setDbInitialized] = useState(false);
  
  // Fetch ledger entries on component mount
  useEffect(() => {
    fetchLedgerEntries();
  }, []);
  
  const fetchLedgerEntries = async () => {
    setIsLoading(true);
    try {
      const watermarks = await dbApi.getWatermarks();
      setEntries(watermarks);
      setDbInitialized(watermarks.length > 0);
    } catch (error) {
      console.error("Error fetching ledger entries:", error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const initializeDatabase = async () => {
    setIsInitializing(true);
    const success = await dbApi.initializeDatabase();
    if (success) {
      await fetchLedgerEntries();
      setDbInitialized(true);
    }
    setIsInitializing(false);
  };
  
  const filteredEntries = entries.filter(entry => {
    // Filter by search term
    const matchesSearch = 
      (entry.meta_data?.filename || "").toLowerCase().includes(searchTerm.toLowerCase()) || 
      entry.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      entry.method.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (entry.purpose || "").toLowerCase().includes(searchTerm.toLowerCase());
      
    // Filter by selected roles (if we have user data)
    const userRole = entry.meta_data?.user_role || "";
    const matchesRole = selectedRoles.length === 0 || selectedRoles.includes(userRole);
    
    return matchesSearch && matchesRole;
  });
  
  const handleRoleToggle = (role: string) => {
    setSelectedRoles(prev => 
      prev.includes(role) 
        ? prev.filter(r => r !== role) 
        : [...prev, role]
    );
  };
  
  const handleExport = () => {
    // Create CSV content
    const headers = ["ID", "Date", "Method", "Action", "Message", "SNR", "Detection Prob", "BER", "Purpose"];
    const csvContent = [
      headers.join(","),
      ...filteredEntries.map(entry => [
        entry.id,
        entry.created_at,
        entry.method,
        entry.action,
        entry.message,
        entry.snr_db,
        entry.detection_probability,
        entry.ber,
        entry.purpose
      ].join(","))
    ].join("\n");
    
    // Create download link
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `watermark_ledger_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    toast.success("Ledger exported successfully");
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
                  placeholder="Search by filename, message or method..."
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
          
          <div className="mt-4 flex justify-end gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={fetchLedgerEntries}
              disabled={isLoading}
            >
              <RefreshCcw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            
            <Button 
              variant="outline" 
              size="sm"
              onClick={initializeDatabase}
              disabled={isInitializing || (dbInitialized && entries.length > 0)}
            >
              <Database className="h-4 w-4 mr-2" />
              {isInitializing ? "Initializing..." : "Initialize Database"}
            </Button>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Watermark History</CardTitle>
        </CardHeader>
        
        <CardContent>
          {isLoading ? (
            // Loading skeleton
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-4">Date</th>
                    <th className="text-left py-3 px-4">File</th>
                    <th className="text-left py-3 px-4">Method</th>
                    <th className="text-left py-3 px-4">Purpose</th>
                    <th className="text-left py-3 px-4">Payload</th>
                    <th className="text-left py-3 px-4">SNR</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEntries.map((entry) => (
                    <tr key={entry.id} className="hover:bg-black/5 dark:hover:bg-white/5">
                      <td className="py-3 px-4">
                        {new Date(entry.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <FileAudio className="h-4 w-4 mr-2 text-muted-foreground" />
                          <span>{entry.meta_data?.filename || "Unknown file"}</span>
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {entry.action === "embed" ? "Watermarked" : "Detection"}
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center rounded-full bg-blue-100 dark:bg-blue-900/30 px-2 py-1 text-xs">
                          {entry.method}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-1 text-xs">
                          {entry.purpose || "general"}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono">
                        {entry.message}
                      </td>
                      <td className="py-3 px-4">
                        {entry.snr_db ? `${entry.snr_db.toFixed(1)} dB` : "N/A"}
                      </td>
                    </tr>
                  ))}
                  
                  {filteredEntries.length === 0 && (
                    <tr>
                      <td colSpan={6} className="py-6 text-center text-muted-foreground">
                        {entries.length === 0 ? (
                          <div>
                            <p>No entries found in the database</p>
                            <Button 
                              variant="link" 
                              onClick={initializeDatabase} 
                              disabled={isInitializing}
                            >
                              Initialize database with sample data
                            </Button>
                          </div>
                        ) : (
                          "No entries match your search criteria"
                        )}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
          
          <div className="mt-4 flex justify-end">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleExport}
              disabled={filteredEntries.length === 0}
            >
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
