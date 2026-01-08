import { memo, useState, useRef, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw, Volume2, Loader2 } from "lucide-react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";

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
  const [isPlaying, setIsPlaying] = useState(false);
  const [editablePrompt, setEditablePrompt] = useState(prompt);
  const [volume, setVolume] = useState(weight_db);
  const audioRef = useRef(null);

  const audioSrc = audioBase64 || audio_base64;
  const durationSec = duration_ms / 1000;
  const fadeSec = fade_ms / 1000;
  const fadePercent = Math.min((fadeSec / durationSec) * 100, 25);

  const typeColors = useMemo(() => ({
    SFX: "bg-primary/20 text-primary border-primary/50",
    Ambience: "bg-secondary/20 text-secondary border-secondary/50",
    Music: "bg-violet-500/20 text-violet-400 border-violet-500/50",
    Dialogue: "bg-orange-500/20 text-orange-400 border-orange-500/50",
  }), []);

  const handlePlayPause = useCallback(() => {
    if (!audioRef.current && audioSrc) {
      audioRef.current = new Audio(`data:audio/mp3;base64,${audioSrc}`);
      audioRef.current.onended = () => setIsPlaying(false);
    }
    
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play().catch(console.error);
      }
      setIsPlaying(!isPlaying);
    } else {
      // Demo mode - toggle visual state
      setIsPlaying(!isPlaying);
    }
  }, [audioSrc, isPlaying]);

  const handleVolumeChange = useCallback((value) => {
    const newVolume = value[0];
    setVolume(newVolume);
    handleUpdate({ weight_db: newVolume });
  }, [handleUpdate]);

  const handleRegenerate = useCallback(() => {
    onRegenerate(id, editablePrompt);
  }, [id, editablePrompt, onRegenerate]);

  const handlePromptChange = useCallback((e) => {
    setEditablePrompt(e.target.value);
  }, []);

  // Generate waveform bars once
  const waveformBars = useMemo(() => 
    Array.from({ length: 40 }, () => Math.random() * 0.6 + 0.3),
  []);

  return (
    <motion.div
      className="glass-panel p-4 border border-border/30 rounded-xl"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header with Badge, Prompt, and Regenerate */}
      <div className="flex items-center gap-3 mb-4">
        <Badge 
          variant="outline" 
          className={`font-display text-[10px] tracking-wider shrink-0 ${typeColors[type] || typeColors.SFX}`}
        >
          {type}
        </Badge>
        
        <input
          type="text"
          value={editablePrompt}
          onChange={handlePromptChange}
          className="flex-1 bg-transparent border-b border-border/50 px-2 py-1 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
          placeholder="Enter audio prompt..."
        />
        
        <Button
          variant="ghost"
          size="sm"
          className="glass-panel shrink-0"
          onClick={handleRegenerate}
          disabled={isRegenerating}
        >
          {isRegenerating ? (
            <Loader2 className="w-4 h-4 animate-spin text-primary" />
          ) : (
            <RotateCcw className="w-4 h-4 text-muted-foreground" />
          )}
          <span className="ml-1 text-xs">{isRegenerating ? "..." : "Regen"}</span>
        </Button>
      </div>

      {/* Waveform with Fade Visualization */}
      <div className="relative h-16 rounded-lg overflow-hidden bg-muted/20">
        {/* Fade-in gradient overlay */}
        <div 
          className="absolute left-0 top-0 bottom-0 z-10 pointer-events-none"
          style={{
            width: `${fadePercent}%`,
            background: "linear-gradient(to right, hsl(var(--background)), transparent)"
          }}
        />
        
        {/* Fade-out gradient overlay */}
        <div 
          className="absolute right-0 top-0 bottom-0 z-10 pointer-events-none"
          style={{
            width: `${fadePercent}%`,
            background: "linear-gradient(to left, hsl(var(--background)), transparent)"
          }}
        />

        {/* Waveform bars */}
        <div className="absolute inset-0 flex items-center justify-around px-1">
          {waveformBars.map((height, i) => {
            const position = i / waveformBars.length;
            const isFadingIn = position < fadePercent / 100;
            const isFadingOut = position > (1 - fadePercent / 100);
            const opacity = isFadingIn 
              ? position / (fadePercent / 100)
              : isFadingOut 
                ? (1 - position) / (fadePercent / 100)
                : 1;
            
            return (
              <div
                key={i}
                className="w-1 rounded-full transition-all duration-300"
                style={{
                  height: `${height * 100}%`,
                  opacity: Math.max(0.2, opacity),
                  background: isPlaying 
                    ? `linear-gradient(to top, hsl(var(--primary) / ${opacity}), hsl(var(--secondary) / ${opacity}))`
                    : `hsl(var(--muted-foreground) / ${opacity * 0.6})`,
                }}
              />
            );
          })}
        </div>

        {/* Playhead animation */}
        {isPlaying && (
          <motion.div
            className="absolute top-0 bottom-0 w-0.5 bg-secondary z-20"
            initial={{ left: "0%" }}
            animate={{ left: "100%" }}
            transition={{ duration: durationSec, ease: "linear", repeat: Infinity }}
          />
        )}
      </div>

      {/* Controls Row */}
      <div className="flex items-center gap-4 mt-4">
        {/* Play Button */}
        <Button
          variant="ghost"
          size="icon"
          className={`h-8 w-8 rounded-full ${isPlaying ? 'bg-primary/20' : 'bg-muted/50'}`}
          onClick={handlePlayPause}
        >
          {isPlaying ? (
            <Pause className="w-4 h-4 text-primary" />
          ) : (
            <Play className="w-4 h-4 text-primary ml-0.5" />
          )}
        </Button>

        {/* Volume Slider */}
        <div className="flex items-center gap-2 flex-1 max-w-[200px]">
          <Volume2 className="w-4 h-4 text-muted-foreground shrink-0" />
          <Slider
            value={[volume]}
            onValueChange={handleVolumeChange}
            min={-20}
            max={6}
            step={0.5}
            className="flex-1"
          />
          <span className="text-xs font-mono text-muted-foreground w-12 text-right">
            {volume > 0 ? '+' : ''}{volume.toFixed(1)}dB
          </span>
        </div>

        {/* Duration Info */}
        <div className="ml-auto text-xs text-muted-foreground font-mono">
          {durationSec.toFixed(1)}s • {fade_ms}ms fade
        </div>
      </div>
    </motion.div>
  );
});

AudioCard.displayName = "AudioCard";

export default AudioCard;
