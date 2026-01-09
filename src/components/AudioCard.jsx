import { motion } from "framer-motion";
import { useState, useEffect, useRef, useCallback, memo } from "react";
import { Play, Pause, RotateCcw, Volume2, Loader2 } from "lucide-react";
import Waveform from "./timeline/Waveform";
import TimelineRuler from "./timeline/TimelineRuler";
import DraggableMarker from "./timeline/DraggableMarker";
import VerticalFader from "./mixer/VerticalFader";
import SpatialPanPad from "./mixer/SpatialPanPad";
import CurveSelector from "./mixer/CurveSelector";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

const AudioCard = memo(({ 
  id,
  type = "SFX",
  prompt = "Thunderstorm with distant rumbling",
  start_time_ms = 0,
  duration_ms = 10000,
  weight_db = 0,
  fade_ms = 500,
  audioBase64 = null,
  audio_base64 = null,
  isRegenerating = false,
  handleUpdate = () => {},
  onRegenerate = () => {},
}) => {
  const audioRef = useRef(null);
  const intervalRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [fadeIn, setFadeIn] = useState(0.1);
  const [fadeOut, setFadeOut] = useState(0.9);
  const [volume, setVolume] = useState(weight_db || 0);
  const [pan, setPan] = useState({ x: 0.5, y: 0.5 });
  const [fadeInCurve, setFadeInCurve] = useState("logarithmic");
  const [fadeOutCurve, setFadeOutCurve] = useState("exponential");
  const [editablePrompt, setEditablePrompt] = useState(prompt);

  // Get audio data from either prop name
  const audioData = audioBase64 || audio_base64;
  
  // Convert duration_ms to seconds for display
  const durationSeconds = duration_ms / 1000;

  // Sync state with props when they change
  useEffect(() => {
    setEditablePrompt(prompt);
    setVolume(weight_db || 0);
  }, [prompt, weight_db]);

  // Initialize audio element when audio data is available
  useEffect(() => {
    // Clean up previous audio if it exists
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    
    if (audioData) {
      const audioUrl = audioData.startsWith('data:') 
        ? audioData 
        : `data:audio/wav;base64,${audioData}`;
      
      audioRef.current = new Audio(audioUrl);
      audioRef.current.volume = Math.max(0, Math.min(1, (volume + 12) / 12)); // Convert dB to 0-1 range
      
      audioRef.current.onended = () => {
        setIsPlaying(false);
        setCurrentTime(0);
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
      
      audioRef.current.ontimeupdate = () => {
        if (audioRef.current) {
          setCurrentTime(audioRef.current.currentTime);
        }
      };
    } else {
      setIsPlaying(false);
      setCurrentTime(0);
    }

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [audioData]); // Only recreate when audioData changes

  // Update audio volume when volume state changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = Math.max(0, Math.min(1, (volume + 12) / 12));
    }
  }, [volume]);

  // Handle play/pause
  const handlePlayPause = useCallback(() => {
    if (audioRef.current && audioData) {
      if (isPlaying) {
        audioRef.current.pause();
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      } else {
        audioRef.current.play().catch(err => {
          console.error("Error playing audio:", err);
          setIsPlaying(false);
        });
      }
      setIsPlaying(!isPlaying);
    }
  }, [isPlaying, audioData]);

  // Handle volume change with update callback
  const handleVolumeChange = useCallback((newVolume) => {
    setVolume(newVolume);
    handleUpdate(id, { weight_db: newVolume });
  }, [id, handleUpdate]);

  // Handle fade in change with update callback
  const handleFadeInChange = useCallback((newFadeIn) => {
    setFadeIn(newFadeIn);
    const fadeInMs = Math.round(newFadeIn * duration_ms);
    handleUpdate(id, { fade_in_ms: fadeInMs });
  }, [id, duration_ms, handleUpdate]);

  // Handle fade out change with update callback
  const handleFadeOutChange = useCallback((newFadeOut) => {
    setFadeOut(newFadeOut);
    const fadeOutMs = Math.round((1 - newFadeOut) * duration_ms);
    handleUpdate(id, { fade_out_ms: fadeOutMs });
  }, [id, duration_ms, handleUpdate]);

  // Handle prompt change with update callback
  const handlePromptChange = useCallback((newPrompt) => {
    setEditablePrompt(newPrompt);
    handleUpdate(id, { audio_class: newPrompt });
  }, [id, handleUpdate]);

  // Handle pan change with update callback
  const handlePanChange = useCallback((newPan) => {
    setPan(newPan);
    handleUpdate(id, { pan_x: newPan.x, pan_y: newPan.y });
  }, [id, handleUpdate]);

  // Handle regenerate
  const handleRegenerate = useCallback(() => {
    if (onRegenerate) {
      onRegenerate(id);
    }
  }, [id, onRegenerate]);

  const typeColors = {
    SFX: "bg-primary/20 text-primary border-primary/50",
    Ambience: "bg-secondary/20 text-secondary border-secondary/50",
    Music: "bg-neon-purple/20 text-neon-purple border-neon-purple/50",
    Dialogue: "bg-orange-500/20 text-orange-400 border-orange-500/50",
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
          className={`font-display text-[10px] tracking-wider ${typeColors[type] || typeColors.SFX}`}
        >
          {type}
        </Badge>
        <input
          type="text"
          value={editablePrompt}
          onChange={(e) => handlePromptChange(e.target.value)}
          onBlur={(e) => handlePromptChange(e.target.value)}
          className="flex-1 bg-transparent border-b border-border/50 px-2 py-1 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
          placeholder="Enter audio prompt..."
        />
        {audioData && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        
        
          <Button 
            type="button"
            variant="outline"
            size="sm"
            className="ml-2 flex items-center gap-1 px-2 py-1 text-xs"
            onClick={handleRegenerate}
            disabled={isRegenerating}
          >
            {isRegenerating ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin mr-1" /> Regenerating
              </>
            ) : (
              <>
                <RotateCcw className="w-3 h-3 mr-1" /> Regenerate
              </>
            )}
          </Button>
          </div>
        )}
      </div>

      <div className="flex gap-4">
        {/* Timeline Section */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-2">
            <TimelineRuler duration={durationSeconds} width={800} />
            {audioData && (
              <div className="text-xs font-mono text-muted-foreground">
                {formatTime(currentTime)} / {formatTime(durationSeconds)}
              </div>
            )}
          </div>
          
          <div className="relative mt-2" style={{ width: '100%', maxWidth: '800px' }}>
            <Waveform 
              width={800} 
              height={80} 
              bars={50}
              isPlaying={isPlaying}
              highlightStart={fadeIn}
              highlightEnd={fadeOut}
            />
            
            {/* Playhead indicator */}
            {audioData && (
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-secondary shadow-[0_0_8px_hsl(var(--secondary))] z-20 pointer-events-none"
                style={{ 
                  left: `${(currentTime / durationSeconds) * 100}%`,
                  display: isPlaying ? 'block' : 'none'
                }}
              />
            )}
            
            {/* Draggable Markers */}
            <DraggableMarker
              position={fadeIn}
              containerWidth={800}
              onPositionChange={handleFadeInChange}
              label="Fade In"
              color="primary"
            />
            <DraggableMarker
              position={fadeOut}
              containerWidth={800}
              onPositionChange={handleFadeOutChange}
              label="Fade Out"
              color="secondary"
            />
          </div>

          {/* Curve Selectors */}
          {/* <div className="flex gap-4 mt-4">
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
          </div> */}
        </div>

        {/* Mixer Section */}
        <div className="flex gap-3 flex-shrink-0">
          <VerticalFader 
            value={volume} 
            onChange={handleVolumeChange}
            label="VOL"
          />
          <SpatialPanPad 
            size={100} 
            x={pan.x}
            y={pan.y}
            onChange={handlePanChange}
          />
        </div>
      </div>

      {/* Action Bar */}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-border/30">
        <Button
          variant="ghost"
          size="sm"
          className={`glass-panel ${isPlaying ? 'neon-glow' : ''} ${!audioData ? 'opacity-50 cursor-not-allowed' : ''}`}
          onClick={handlePlayPause}
          disabled={!audioData}
          title={audioData ? (isPlaying ? 'Pause' : 'Play') : 'No audio loaded'}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4 text-primary" />
          ) : (
            <Play className="w-4 h-4 text-primary" />
          )}
        </Button>
        
        
        <div className="flex-1" />
        
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Volume2 className="w-3 h-3" />
            <span className="font-mono">{volume > 0 ? '+' : ''}{volume.toFixed(1)} dB</span>
          </div>
          {audioData && (
            <div className="flex items-center gap-1 pl-2 border-l border-border/30">
              <span className="text-[10px]">Start:</span>
              <span className="font-mono">{formatTime(start_time_ms / 1000)}</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
});

AudioCard.displayName = "AudioCard";

export default AudioCard;