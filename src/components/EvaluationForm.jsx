import { memo, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { 
  UserCheck, 
  ArrowRight,
  Save,
  CheckCircle2,
  Star
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
import useAudioStore from "../store/useAudioStore";

const automatedMetrics = [
  { name: "CLAP Score", value: 0.92, target: 0.85, description: "Audio-text alignment" },
  { name: "Spectral Richness", value: 0.78, target: 0.70, description: "Frequency distribution" },
  { name: "Dynamic Range", value: 0.85, target: 0.80, description: "Loudness variation" },
  { name: "Noise Floor", value: -60, target: -50, description: "dB below signal", isLower: true },
];

const EvaluationForm = memo(({ onSave }) => {
  const [step, setStep] = useState("form");
  const [personName, setPersonName] = useState("");
  const [humanScores, setHumanScores] = useState({
    dramatization: 0,
    syncAccuracy: 0,
    atmosphericDepth: 0,
  });
  const [feedback, setFeedback] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  
  const saveResults = useAudioStore(state => state.saveResults);

  const handleScoreChange = useCallback((key, value) => {
    setHumanScores(prev => ({ ...prev, [key]: value }));
  }, []);

  const isFormValid = personName.trim() && Object.values(humanScores).every(v => v > 0);

  const handleSubmit = useCallback(() => {
    setStep("results");
  }, []);

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    const evaluationData = {
      personName,
      humanScores,
      feedback,
      timestamp: new Date().toISOString()
    };
    
    await saveResults(evaluationData);
    onSave?.(evaluationData);
    setIsSaving(false);
  }, [personName, humanScores, feedback, saveResults, onSave]);

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
      className="glass-panel p-6 rounded-xl border border-border/30"
    >
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
          <h4 className="text-2xl font-bold text-emerald-400">8.4 / 10</h4>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setStep("form")}>
            Re-evaluate
          </Button>
          <Button size="sm" onClick={handleSave} disabled={isSaving}>
            <Save className="w-4 h-4 mr-2" />
            {isSaving ? "Saving..." : "Save Results"}
          </Button>
        </div>
      </div>
    </motion.div>
  );
});

EvaluationForm.displayName = "EvaluationForm";

export default EvaluationForm;
