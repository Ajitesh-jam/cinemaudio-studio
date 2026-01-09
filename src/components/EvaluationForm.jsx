import { memo, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { 
  UserCheck, 
  ArrowRight,
  Save,
  CheckCircle2,
  Star,
  Loader2
} from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Input } from "./ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";


const automatedMetrics = [
  { name: "CLAP Score", value: 0.92, target: 0.85, description: "Audio-text alignment" },
  { name: "Spectral Richness", value: 0.78, target: 0.70, description: "Frequency distribution" },
  { name: "Dynamic Range", value: 0.85, target: 0.80, description: "Loudness variation" },
  { name: "Noise Floor", value: -60, target: -50, description: "dB below signal", isLower: true },
];

// Helper function to extract Google Drive folder ID from URL
const extractDriveFolderId = (link) => {
  if (!link) return null;
  
  // Handle direct ID
  if (!link.includes('http') && !link.includes('/')) {
    return link;
  }
  
  // Handle full URL: https://drive.google.com/drive/folders/FOLDER_ID
  const foldersMatch = link.match(/\/folders\/([a-zA-Z0-9_-]+)/);
  if (foldersMatch) {
    return foldersMatch[1];
  }
  
  // Handle short URL or other formats
  const idMatch = link.match(/([a-zA-Z0-9_-]{25,})/);
  if (idMatch) {
    return idMatch[1];
  }
  
  return null;
};

// Helper function to extract Google Sheets ID from URL
const extractSheetId = (link) => {
  if (!link) return null;
  
  // Handle direct ID
  if (!link.includes('http') && !link.includes('/')) {
    return link;
  }
  
  // Handle full URL: https://docs.google.com/spreadsheets/d/SHEET_ID/edit
  const spreadsheetsMatch = link.match(/\/spreadsheets\/d\/([a-zA-Z0-9_-]+)/);
  if (spreadsheetsMatch) {
    return spreadsheetsMatch[1];
  }
  
  // Handle short URL or other formats
  const idMatch = link.match(/([a-zA-Z0-9_-]{25,})/);
  if (idMatch) {
    return idMatch[1];
  }
  
  return null;
};

const EvaluationForm = memo(({ audioBase64, storyText, driveFolderLink, sheetLink }) => {
  const [step, setStep] = useState("form");
  const [personName, setPersonName] = useState("");
  const [humanScores, setHumanScores] = useState({
    dramatization: 0,
    syncAccuracy: 0,
    atmosphericDepth: 0,
  });
  const [feedback, setFeedback] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null); // null, 'saving', 'success', 'error'

  const handleScoreChange = useCallback((key, value) => {
    setHumanScores(prev => ({ ...prev, [key]: value }));
  }, []);

  const isFormValid = personName.trim() && Object.values(humanScores).every(v => v > 0);

  const handleSubmit = useCallback(() => {
    setStep("results");
  }, []);

  // Calculate final score from human scores (average * 2 to get 0-10 scale)
  const calculateFinalScore = useCallback(() => {
    const avg = Object.values(humanScores).reduce((sum, val) => sum + val, 0) / Object.values(humanScores).length;
    return (avg / 5) * 10; // Convert 0-5 scale to 0-10 scale
  }, [humanScores]);

  
  const handleSave = useCallback(async () => {
    if (!audioBase64) {
      alert("No audio data available to save.");
      return;
    }

    // Extract folder ID and sheet ID from links
    const driveFolderId = extractDriveFolderId(driveFolderLink);
    const sheetId = extractSheetId(sheetLink);

    // Validate that IDs were extracted correctly
    if (!driveFolderId) {
      alert("Error: Could not extract Google Drive folder ID. Please check the folder link.");
      return;
    }

    if (!sheetId) {
      alert("Error: Could not extract Google Sheets ID. Please check the sheet link.");
      return;
    }

    setIsSaving(true);
    setSaveStatus('saving');
    
    try {
      // Convert base64 string to Blob
      const base64Data = audioBase64.includes(',') 
        ? audioBase64.split(',')[1] 
        : audioBase64;
      
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const audioBlob = new Blob([byteArray], { type: 'audio/wav' });

      // Convert Blob to Base64 for Google Apps Script
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      
      reader.onloadend = async () => {
        try {
          const base64Audio = reader.result;
          
          // Calculate automated metrics (using placeholder values for now)
          const autoMetrics = {
            clapScore: 0.92,
            spectralRichness: 0.78,
            dynamicRange: 0.85,
            noiseFloor: -60
          };
          
          const finalScore = calculateFinalScore();
    
          const payload = {
            personName,
            humanScores, // { dramatization, syncAccuracy, atmosphericDepth }
            feedback,
            autoMetrics, // { clapScore, spectralRichness, etc. }
            finalScore: finalScore.toFixed(1),
            storyPrompt: storyText || "N/A",
            audioFile: base64Audio,
            timestamp: new Date().toLocaleString(),
            driveFolderId, // Google Drive folder ID
            sheetId // Google Sheets ID
          };
    
          // Send to Google Apps Script which will handle:
          // 1. Uploading audio file to Google Drive folder
          // 2. Writing evaluation data to Google Sheets
          await fetch("https://script.google.com/macros/s/AKfycbyM1tNf3KQ7CPtgksGODLSO7xnPp3xgAt9DcfQUZ-pRZ3DQqb7jRElay2K4-MVn16XO/exec", {
            method: "POST",
            mode: "no-cors", // Required for cross-origin GAS requests
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
    
          // Since mode is 'no-cors', we won't get a readable response body, 
          // but we can assume success if no error is thrown.
          setSaveStatus('success');
          setTimeout(() => {
            alert("Results saved successfully to Drive and Sheets!");
            setSaveStatus(null);
          }, 500);
        } catch (error) {
          console.error("Save failed:", error);
          setSaveStatus('error');
          setTimeout(() => {
            alert("Error saving data.");
            setSaveStatus(null);
          }, 500);
        } finally {
          setIsSaving(false);
        }
      };
      
      reader.onerror = () => {
        console.error("FileReader error");
        setSaveStatus('error');
        setIsSaving(false);
        setTimeout(() => {
          alert("Error processing audio file.");
          setSaveStatus(null);
        }, 500);
      };
    } catch (error) {
      console.error("Save failed:", error);
      setSaveStatus('error');
      setIsSaving(false);
      setTimeout(() => {
        alert("Error saving data.");
        setSaveStatus(null);
      }, 500);
    }
  }, [personName, humanScores, feedback, audioBase64, storyText, calculateFinalScore, driveFolderLink, sheetLink]);



  const getStatusColor = (value, target, isLower = false) => {
    const passed = isLower ? value <= target : value >= target;
    return passed ? "text-emerald-400" : "text-rose-400";
  };

  if (step === "form") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel p-6 rounded-xl border border-border/30"
      >
        <div className="flex items-center gap-3 mb-6">
          <UserCheck className="w-5 h-5 text-primary" />
          <div>
            <h2 className="text-lg font-display font-bold text-foreground">Perceptual Evaluation</h2>
            <p className="text-sm text-muted-foreground">Rate the generated audio</p>
          </div>
        </div>

        {/* Person Name */}
        <div className="mb-6">
          <label className="text-sm font-medium text-foreground mb-2 block">Your Name</label>
          <Input
            value={personName}
            onChange={(e) => setPersonName(e.target.value)}
            placeholder="Enter your name..."
            className="bg-muted/30"
          />
        </div>

        {/* Rating Questions */}
        <div className="space-y-6">
          {[
            { id: "dramatization", label: "Dramatization", sub: "Emotional capture" },
            { id: "syncAccuracy", label: "Sync Accuracy", sub: "Timeline alignment" },
            { id: "atmosphericDepth", label: "Atmospheric Depth", sub: "Spatial immersion" },
          ].map((q) => (
            <div key={q.id}>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-foreground">{q.label}</label>
                <span className="text-xs text-muted-foreground">{q.sub}</span>
              </div>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map((num) => (
                  <button
                    key={num}
                    onClick={() => handleScoreChange(q.id, num)}
                    className={`flex-1 py-2 rounded-md border transition-all ${
                      humanScores[q.id] === num
                        ? "bg-primary border-primary text-primary-foreground"
                        : "bg-muted/30 border-border/50 text-muted-foreground hover:border-primary/50"
                    }`}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Feedback */}
        <div className="mt-6">
          <label className="text-sm font-medium text-foreground mb-2 block">
            Suggestions <span className="text-muted-foreground">(Optional)</span>
          </label>
          <Textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Any improvement ideas..."
            className="bg-muted/30 min-h-[80px]"
          />
        </div>

        <Button
          disabled={!isFormValid}
          onClick={handleSubmit}
          className="w-full mt-6"
        >
          View Results
          <ArrowRight className="ml-2 w-4 h-4" />
        </Button>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel p-6 rounded-xl border border-border/30 relative"
    >
      {/* Saving Overlay */}
      {isSaving && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-background/80 backdrop-blur-sm rounded-xl z-50 flex flex-col items-center justify-center gap-4"
        >
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
          <div className="text-center">
            <p className="font-display text-lg text-foreground mb-1">Saving Results</p>
            <p className="text-sm text-muted-foreground">Uploading to Google Drive and Sheets...</p>
          </div>
        </motion.div>
      )}
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display text-lg tracking-wider text-foreground">
          EVALUATION RESULTS
        </h3>
        <span className="text-xs font-mono text-muted-foreground">
          Evaluator: {personName}
        </span>
      </div>

      <Table>
        <TableHeader>
          <TableRow className="border-border/30">
            <TableHead className="text-muted-foreground text-xs">SOURCE</TableHead>
            <TableHead className="text-muted-foreground text-xs">METRIC</TableHead>
            <TableHead className="text-muted-foreground text-xs text-right">VALUE</TableHead>
            <TableHead className="text-muted-foreground text-xs text-center">STATUS</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Object.entries(humanScores).map(([key, val]) => (
            <TableRow key={key} className="border-border/20">
              <TableCell>
                <span className="text-[10px] bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded">Human</span>
              </TableCell>
              <TableCell className="capitalize text-foreground text-sm">
                {key.replace(/([A-Z])/g, ' $1')}
              </TableCell>
              <TableCell className="text-right font-mono text-amber-400">{val}/5</TableCell>
              <TableCell className="text-center">
                {val >= 4 ? <CheckCircle2 className="w-4 h-4 text-emerald-400 mx-auto" /> : <Star className="w-4 h-4 text-muted-foreground mx-auto" />}
              </TableCell>
            </TableRow>
          ))}
          {automatedMetrics.map((metric) => (
            <TableRow key={metric.name} className="border-border/20">
              <TableCell>
                <span className="text-[10px] bg-primary/10 text-primary px-2 py-0.5 rounded">Auto</span>
              </TableCell>
              <TableCell className="text-foreground text-sm">{metric.name}</TableCell>
              <TableCell className={`text-right font-mono ${getStatusColor(metric.value, metric.target, metric.isLower)}`}>
                {metric.value < 1 && metric.value > -1
                  ? (metric.value * 100).toFixed(0) + '%'
                  : metric.value + (metric.isLower ? ' dB' : '')}
              </TableCell>
              <TableCell className="text-center">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 mx-auto" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {/* Final Score */}
      <div className="mt-6 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg flex items-center justify-between">
        <div>
          <p className="text-xs text-emerald-400/70 uppercase tracking-wider">Final Score</p>
          <h4 className="text-2xl font-bold text-emerald-400">
            {calculateFinalScore().toFixed(1)} / 10
          </h4>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setStep("form")}
            disabled={isSaving}
          >
            Re-evaluate
          </Button>
          <Button 
            size="sm" 
            onClick={handleSave} 
            disabled={isSaving}
            className={saveStatus === 'success' ? 'bg-emerald-500 hover:bg-emerald-600' : ''}
          >
            {isSaving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : saveStatus === 'success' ? (
              <>
                <CheckCircle2 className="w-4 h-4 mr-2" />
                Saved!
              </>
            ) : saveStatus === 'error' ? (
              <>
                <Save className="w-4 h-4 mr-2" />
                Retry Save
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Results
              </>
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  );
});

EvaluationForm.displayName = "EvaluationForm";

export default EvaluationForm;
