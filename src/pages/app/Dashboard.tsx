import { useState, useContext } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import { Upload, FileAudio, Loader2, Info, CheckCircle, ShieldAlert } from "lucide-react";
import { api, PreEmbedDetectResponse, WatermarkEmbedResponse, ProcessAudioParams } from "@/services/api";
import { AuthContext } from "@/contexts/AuthContext";
import { userRegistry } from "@/services/userRegistry";

// --- Mock data for UI development ---
const mockPurposes = ["Training", "Internal Review", "Remix", "Commercial Release"];

// --- Helper Interfaces ---
interface DetectedWatermark {
  step: number;
  payload: string;
  userId: number;
  purpose: string;
}

const Dashboard = () => {
  const { user: authenticatedUser } = useContext(AuthContext);
  
  // --- STATE MANAGEMENT ---
  // File and selection state
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioName, setAudioName] = useState("");
  const [selectedPurpose, setSelectedPurpose] = useState(mockPurposes[2]);

  // Loading states
  const [isCheckingFile, setIsCheckingFile] = useState(false);
  const [isEmbedding, setIsEmbedding] = useState(false);

  // Result states - THIS IS THE "MEMORY" OF THE COMPONENT
  const [preEmbedResult, setPreEmbedResult] = useState<PreEmbedDetectResponse | null>(null);
  const [detectedWatermarks, setDetectedWatermarks] = useState<DetectedWatermark[]>([]);
  const [finalEmbedResult, setFinalEmbedResult] = useState<WatermarkEmbedResponse | null>(null);

  // --- HELPER FUNCTIONS ---
  const parseDetectionsForDisplay = (detectedBands: (string | number)[][]): DetectedWatermark[] => {
    const parsed: DetectedWatermark[] = [];
    if (!detectedBands) return parsed;
    for (let i = 0; i < detectedBands.length; i++) {
        const bandBits = detectedBands[i].join('');
        if (bandBits !== '00000000') {
            const userId = parseInt(bandBits.slice(0, 4), 2) || (i + 1);
            const purposeBits = bandBits.slice(4, 6);
            const purposeMap: { [key: string]: string } = {
                '00': 'training', '01': 'internal review',
                '10': 'remix', '11': 'commercial release'
            };
            const purpose = purposeMap[purposeBits] || 'unknown';
            parsed.push({ step: i, payload: bandBits, userId, purpose });
        }
    }
    return parsed;
  };


// In src/pages/app/Dashboard.tsx

const generatePayload = (nextIndex: number) => {
  if (!authenticatedUser) {
      toast.error("User not authenticated.");
      return { message: '00000000', fullCumulativeMessage: '0'.repeat(32) };
  }

  const purposeIndex = mockPurposes.indexOf(selectedPurpose);

  // No more modulo needed! The ID from AuthContext is now small and correct.
  const userBits = authenticatedUser.id.toString(2).padStart(4, '0');
  const purposeBits = purposeIndex.toString(2).padStart(2, '0');
  const randomBits = Math.floor(Math.random() * 4).toString(2).padStart(2, '0');
  const message = `${userBits}${purposeBits}${randomBits}`;

  // The rest of the logic remains the same
  const payloads = Array(4).fill('00000000');
  detectedWatermarks.forEach(wm => {
    if (wm.step < 4) {
      payloads[wm.step] = wm.payload;
    }
  });
  if (nextIndex < 4) {
    payloads[nextIndex] = message;
  }
  const fullCumulativeMessage = payloads.join('');
  
  return { message, fullCumulativeMessage };
};

  // --- CORE LOGIC HANDLERS ---
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      // 1. Reset all state when a new file is uploaded
      setAudioFile(files[0]);
      setAudioName(files[0].name);
      setFinalEmbedResult(null);
      setPreEmbedResult(null);
      setDetectedWatermarks([]);
      setIsCheckingFile(true);

      try {
        // 2. Call the pre-embed detection endpoint to analyze the file
        toast.info("Analyzing file for existing watermarks...");
        const result = await api.preEmbedDetect(files[0]);
        
        // 3. Store the analysis result in state - This is the crucial "memory" step
        setPreEmbedResult(result);
        const foundWatermarks = parseDetectionsForDisplay(result.detected_per_band);
        setDetectedWatermarks(foundWatermarks);
        
        if (foundWatermarks.length > 0) {
            toast.success(`Analysis complete. Found ${foundWatermarks.length} existing watermark(s).`);
        } else {
            toast.success("Analysis complete. This is a clean audio file.");
        }

      } catch (error: any) {
        console.error("Error during file analysis:", error);
        toast.error(error.message || "Failed to analyze audio file.");
        setAudioFile(null); // Clear file on error
      } finally {
        setIsCheckingFile(false);
      }
    }
  };

  const handleEmbedWatermark = async () => {
    // 1. Check if the file has been uploaded and analyzed first
    if (!audioFile || !preEmbedResult) {
      toast.error("Please select a file and wait for analysis to complete.");
      return;
    }

    // 2. Check if all watermark slots are already full
    if (preEmbedResult.next_wm_idx >= 4) {
      toast.error("All 4 watermark slots are full. Cannot embed a new one.");
      return;
    }

    setIsEmbedding(true);
    setFinalEmbedResult(null);

    // 3. Generate the new 8-bit message and the cumulative 32-bit message
    const { message, fullCumulativeMessage } = generatePayload(preEmbedResult.next_wm_idx);

    try {
      const params: ProcessAudioParams = {
        audioFile,
        action: 'embed',
        method: 'pca',
        message,
        full_cumulative_message: fullCumulativeMessage,
        original_audio_file_id: preEmbedResult.original_audio_file_id,
        current_wm_idx: preEmbedResult.next_wm_idx,
        purpose: selectedPurpose.toLowerCase().replace(' ', '_'),
      };

      // 4. Call the API
      const result = await api.processAudio(params);

      // 5. --- THIS IS THE CORRECTED FIX ---
      // The type guard 'if' statement is now correctly included to check the response type.
      if (result && result.action === 'embed') {
        // Inside this 'if' block, TypeScript knows 'result' is a WatermarkEmbedResponse
        setFinalEmbedResult(result as WatermarkEmbedResponse);
        toast.success(`Watermark #${preEmbedResult.next_wm_idx + 1} embedded successfully!`);
        toast.info("You can now download the new file and re-upload it to continue the chain.");
      } else {
        // This handles any unexpected response from the server
        toast.error("Received an unexpected response type after embedding.");
        console.error("Unexpected response from backend:", result);
      }

    } catch (error: any) {
      console.error("Error embedding watermark:", error);
      toast.error(error.message || "Failed to embed watermark.");
    } finally {
      setIsEmbedding(false);
    }
  };

  // --- JSX RENDER ---
  return (
    <div className="max-w-4xl mx-auto pt-12 grid grid-cols-1 md:grid-cols-2 gap-8">
      {/* Left Column: Uploader and Controls */}
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>1. Upload Audio</CardTitle>
          </CardHeader>
          <CardContent>
            <Label htmlFor="audio-upload-embed" className="border-2 border-dashed rounded-lg p-6 text-center block cursor-pointer">
              <Input id="audio-upload-embed" type="file" accept="audio/*" onChange={handleFileChange} className="hidden" />
              {audioFile ? (
                <>
                  <FileAudio className="h-12 w-12 text-primary mx-auto mb-2" />
                  <p className="font-medium">{audioName}</p>
                  <p className="text-xs text-muted-foreground mt-1">Click to change file</p>
                </>
              ) : (
                <>
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                  <p className="font-medium">Click to upload audio</p>
                  <p className="text-xs text-muted-foreground mt-1">WAV, MP3, FLAC supported</p>
                </>
              )}
            </Label>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>2. Configure New Watermark</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {authenticatedUser && (
              <div className="p-3 bg-muted rounded-md">
                <p className="text-sm font-medium">Current User: {authenticatedUser.name || authenticatedUser.email}</p>
                <p className="text-xs text-muted-foreground">Role: {authenticatedUser.role.replace('_', ' ')}</p>
                <p className="text-xs text-muted-foreground">ID: {authenticatedUser.id}</p>
              </div>
            )}
            <div>
              <Label>Purpose</Label>
              <Select onValueChange={setSelectedPurpose} defaultValue={selectedPurpose}>
                <SelectTrigger><SelectValue placeholder="Select a purpose" /></SelectTrigger>
                <SelectContent>
                  {mockPurposes.map(purpose => <SelectItem key={purpose} value={purpose}>{purpose}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Button onClick={handleEmbedWatermark} className="w-full" disabled={isCheckingFile || isEmbedding || !audioFile || finalEmbedResult !== null}>
          {(isCheckingFile || isEmbedding) && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {isCheckingFile ? "Analyzing File..." : isEmbedding ? "Embedding Watermark..." : `Embed Watermark #${(preEmbedResult?.next_wm_idx ?? 0) + 1}`}
        </Button>
      </div>

      {/* Right Column: Results and Status */}
      <div className="space-y-6">
        <Card>
            <CardHeader>
                <CardTitle>Analysis & Status</CardTitle>
            </CardHeader>
            <CardContent>
                {/* Pre-Embed Analysis Results */}
                <div className="space-y-4">
                    <h3 className="font-semibold text-lg">Existing Watermarks</h3>
                    {isCheckingFile && <div className="flex items-center text-muted-foreground"><Loader2 className="mr-2 h-4 w-4 animate-spin"/>Analyzing...</div>}
                    {!isCheckingFile && detectedWatermarks.length > 0 && (
                        <ul className="space-y-2 text-sm list-disc list-inside">
                            {detectedWatermarks.map((wm) => (
                                <li key={wm.step}>
                                    <span className="font-bold">Band {wm.step}:</span> {userRegistry.getUserName(wm.userId)} ({wm.userId}), Purpose: {wm.purpose}
                                </li>
                            ))}
                        </ul>
                    )}
                    {!isCheckingFile && preEmbedResult && detectedWatermarks.length === 0 && (
                        <div className="flex items-center text-green-600"><CheckCircle className="mr-2 h-4 w-4"/>No existing watermarks found. Ready to embed.</div>
                    )}
                    {!preEmbedResult && !isCheckingFile && (
                        <div className="flex items-center text-muted-foreground"><Info className="mr-2 h-4 w-4"/>Upload an audio file to begin.</div>
                    )}
                    {preEmbedResult?.next_wm_idx === 4 && (
                        <div className="flex items-center text-red-600"><ShieldAlert className="mr-2 h-4 w-4"/>All watermark slots are full.</div>
                    )}
                </div>

                <hr className="my-6"/>

                {/* Final Embed Results */}
                <div className="space-y-4">
                    <h3 className="font-semibold text-lg">Latest Embedding Result</h3>
                    {isEmbedding && <div className="flex items-center text-muted-foreground"><Loader2 className="mr-2 h-4 w-4 animate-spin"/>Embedding...</div>}
                    {finalEmbedResult && finalEmbedResult.status === 'success' && (
                        <div className="text-sm space-y-2">
                           <p><span className="font-semibold">Status:</span> Watermark embedded successfully!</p>
                           <p><span className="font-semibold">SNR:</span> {finalEmbedResult.results.snr_db.toFixed(2)} dB</p>
                           {/* <p><span className="font-semibold">MSE:</span> {finalEmbedResult.results.mse.toExponential(4)}</p> */}
                           <p><span className="font-semibold">Bit Error Rate (BER):</span> {finalEmbedResult.results.ber}</p>
                           <p><a href={finalEmbedResult.processed_audio_url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Download Watermarked Audio</a></p>
                        </div>
                    )}
                    {!finalEmbedResult && !isEmbedding && (
                        <div className="flex items-center text-muted-foreground"><Info className="mr-2 h-4 w-4"/>Awaiting embedding process.</div>
                    )}
                </div>

                {/* Debug Section - Remove in production */}
                <hr className="my-6"/>
                <div className="space-y-4">
                    <h3 className="font-semibold text-lg text-muted-foreground">Debug: Registered Users</h3>
                    <div className="text-xs space-y-1">
                        {userRegistry.getAllUsers().map(user => (
                            <div key={user.id} className="flex justify-between">
                                <span>ID {user.id}: {user.name}</span>
                                <span className="text-muted-foreground">{user.role}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;