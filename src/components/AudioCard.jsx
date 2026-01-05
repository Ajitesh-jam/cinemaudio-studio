import { motion } from "framer-motion";
import { useState } from "react";
import { Play, Pause, RotateCcw, Volume2 } from "lucide-react";
import Waveform from "./timeline/Waveform";
import TimelineRuler from "./timeline/TimelineRuler";
import DraggableMarker from "./timeline/DraggableMarker";
import VerticalFader from "./mixer/VerticalFader";
import SpatialPanPad from "./mixer/SpatialPanPad";
import CurveSelector from "./mixer/CurveSelector";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

const AudioCard = ({ 
  id,
  type = "SFX",
  prompt = "Thunderstorm with distant rumbling",
  duration = 10,
  onUpdate 
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [fadeIn, setFadeIn] = useState(0.1);
  const [fadeOut, setFadeOut] = useState(0.9);
  const [volume, setVolume] = useState(0);
  const [fadeInCurve, setFadeInCurve] = useState("logarithmic");
  const [fadeOutCurve, setFadeOutCurve] = useState("exponential");
  const [editablePrompt, setEditablePrompt] = useState(prompt);

  const typeColors = {
    SFX: "bg-primary/20 text-primary border-primary/50",
    Ambience: "bg-secondary/20 text-secondary border-secondary/50",
    Music: "bg-neon-purple/20 text-neon-purple border-neon-purple/50",
    Dialogue: "bg-orange-500/20 text-orange-400 border-orange-500/50",
  };

  return (
    <motion.div
      className="glass-panel p-4 gradient-border"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileHover={{ scale: 1.01 }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <Badge 
          variant="outline" 
          className={`font-display text-[10px] tracking-wider ${typeColors[type]}`}
        >
          {type}
        </Badge>
        <input
          type="text"
          value={editablePrompt}
          onChange={(e) => setEditablePrompt(e.target.value)}
          className="flex-1 bg-transparent border-b border-border/50 px-2 py-1 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
          placeholder="Enter audio prompt..."
        />
      </div>

      <div className="flex gap-4">
        {/* Timeline Section */}
        <div className="flex-1">
          <TimelineRuler duration={duration} width={1000} />
          
          <div className="relative mt-2">
            <Waveform 
              width={1000} 
              height={80} 
              bars={50}
              isPlaying={isPlaying}
              highlightStart={fadeIn}
              highlightEnd={fadeOut}
            />
            
            {/* Draggable Markers */}
            <DraggableMarker
              position={fadeIn}
              containerWidth={1400}
              onPositionChange={setFadeIn}
              label="Fade In"
              color="primary"
            />
            <DraggableMarker
              position={fadeOut}
              containerWidth={400}
              onPositionChange={setFadeOut}
              label="Fade Out"
              color="secondary"
            />
          </div>

          {/* Curve Selectors */}
          <div className="flex gap-4 mt-4">
            <CurveSelector 
              value={fadeInCurve} 
              onChange={setFadeInCurve}
              type="fade-in"
            />
            <CurveSelector 
              value={fadeOutCurve} 
              onChange={setFadeOutCurve}
              type="fade-out"
            />
          </div>
        </div>

        {/* Mixer Section */}
        <div className="flex gap-3">
          <VerticalFader 
            value={volume} 
            onChange={setVolume}
            label="VOL"
          />
          <SpatialPanPad size={100} />
        </div>
      </div>

      {/* Action Bar */}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-border/30">
        <Button
          variant="ghost"
          size="sm"
          className={`glass-panel ${isPlaying ? 'neon-glow' : ''}`}
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4 text-primary" />
          ) : (
            <Play className="w-4 h-4 text-primary" />
          )}
        </Button>
        
        <Button variant="ghost" size="sm" className="glass-panel">
          <RotateCcw className="w-4 h-4 text-muted-foreground" />
          <span className="ml-1 text-xs">Regenerate</span>
        </Button>
        
        <div className="flex-1" />
        
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Volume2 className="w-3 h-3" />
          <span className="font-mono">{volume > 0 ? '+' : ''}{volume.toFixed(1)} dB</span>
        </div>
      </div>
    </motion.div>
  );
};

export default AudioCard;
