import { motion } from "framer-motion";
import { CheckCircle2, AlertCircle, TrendingUp } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";

const metrics = [
  { 
    name: "CLAP Score", 
    value: 0.92, 
    target: 0.85, 
    description: "Audio-text alignment" 
  },
  { 
    name: "Spectral Richness", 
    value: 0.78, 
    target: 0.70, 
    description: "Frequency distribution" 
  },
  { 
    name: "Dynamic Range", 
    value: 0.85, 
    target: 0.80, 
    description: "Loudness variation" 
  },
  { 
    name: "Temporal Coherence", 
    value: 0.88, 
    target: 0.75, 
    description: "Timeline consistency" 
  },
  { 
    name: "Noise Floor", 
    value: -60, 
    target: -50, 
    description: "dB below signal",
    isLower: true 
  },
];

const EvaluationGrid = () => {
  const getStatusColor = (value, target, isLower = false) => {
    const passed = isLower ? value <= target : value >= target;
    return passed ? "text-secondary" : "text-destructive";
  };

  const getStatusIcon = (value, target, isLower = false) => {
    const passed = isLower ? value <= target : value >= target;
    return passed ? (
      <CheckCircle2 className="w-4 h-4 text-secondary" />
    ) : (
      <AlertCircle className="w-4 h-4 text-destructive" />
    );
  };

  return (
    <motion.div
      className="glass-panel p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-4 h-4 text-primary" />
        <h3 className="font-display text-sm tracking-wider text-foreground">
          QUALITY METRICS
        </h3>
      </div>

      <Table>
        <TableHeader>
          <TableRow className="border-border/30">
            <TableHead className="text-muted-foreground font-display text-xs">METRIC</TableHead>
            <TableHead className="text-muted-foreground font-display text-xs text-right">VALUE</TableHead>
            <TableHead className="text-muted-foreground font-display text-xs text-right">TARGET</TableHead>
            <TableHead className="text-muted-foreground font-display text-xs text-center">STATUS</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {metrics.map((metric, index) => (
            <motion.tr
              key={metric.name}
              className="border-border/20 hover:bg-muted/20 transition-colors"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <TableCell>
                <div>
                  <span className="text-foreground text-sm">{metric.name}</span>
                  <p className="text-[10px] text-muted-foreground">{metric.description}</p>
                </div>
              </TableCell>
              <TableCell className={`text-right font-mono text-sm ${getStatusColor(metric.value, metric.target, metric.isLower)}`}>
                {typeof metric.value === 'number' && metric.value < 1 
                  ? (metric.value * 100).toFixed(0) + '%'
                  : metric.value + (metric.isLower ? ' dB' : '')}
              </TableCell>
              <TableCell className="text-right font-mono text-sm text-muted-foreground">
                {typeof metric.target === 'number' && metric.target < 1 
                  ? '≥' + (metric.target * 100).toFixed(0) + '%'
                  : (metric.isLower ? '≤' : '≥') + metric.target + (metric.isLower ? ' dB' : '')}
              </TableCell>
              <TableCell className="text-center">
                {getStatusIcon(metric.value, metric.target, metric.isLower)}
              </TableCell>
            </motion.tr>
          ))}
        </TableBody>
      </Table>
    </motion.div>
  );
};

export default EvaluationGrid;
