import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import HeroSection from "../components/HeroSection";
import AudioCard from "../components/AudioCard";
import MasterMixButton from "../components/MasterMixButton";
import EvaluationGrid from "../components/EvaluationGrid";
import FloatingParticles from "../components/FloatingParticles";
import AnimatedLoader from "../components/ui/AnimatedLoader";

const sampleAudioCues = [
  {
    id: 1,
    type: "Ambience",
    prompt: "Heavy rain with distant thunder rolling",
    duration: 10,
    audioBase64:null
  },
  {
    id: 2,
    type: "SFX",
    prompt: "Car engine starting, tires on gravel",
    duration: 10,
    audioBase64:null
  },
  {
    id: 3,
    type: "Music",
    prompt: "Tense orchestral underscore, low strings",
    duration: 10,
    audioBase64:null
  },
];

const Index = () => {
  const [audioCues, setAudioCues] = useState([]);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleDecompose = (storyText) => {

    // INSERT_YOUR_CODE
    const backendEndpoint = import.meta.env.VITE_BACKEND_ENDPOINT || "http://localhost:8000";
    setIsLoading(true);
    fetch(`${backendEndpoint}/api/v1/decide-cues`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        story_text: storyText,
        speed_wps: 2
      })
    })
    .then(async (response) => {
      if (!response.ok) throw new Error("Failed to fetch audio cues");
      return response.json();
    })
    .then((data) => {
      // Map cues to AudioCard-like objects if needed; otherwise, just use cues as-is

      console.log("data :", data);


      const cues = data?.cues || [];
      // For demonstration, wrap in default structure expected by AudioCard
      const mappedCues = cues.map((cue, idx) => ({
        id: idx + 1,
        type: cue.audio_type || "SFX",
        prompt: cue.audio_class || "",
        duration: cue.duration_ms || 10,
        audioBase64: cue.audio_base64 || null
      }));

      console.log("cues :", cues);

      // INSERT_YOUR_CODE
      // Step 1: Call the /api/v1/generate-audio API on the backend
      fetch(`${backendEndpoint}/api/v1/generate-audio`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          cues: [
            {
              audio_class: "dog barking",
              audio_type: "SFX",
              start_time_ms: 0,
              duration_ms: 1000,
              weight_db: 0,
              fade_ms: 500
            }
          ],
          total_duration_ms: data?.total_duration_ms ?? 1000
        })
      })
        .then(async (res) => {
          if (!res.ok) throw new Error("Failed to generate audio");
          return await res.json();
        })
        .then((audioData) => {
          // audioData: { audio_base64, duration_ms, message }
          // Attach base64 audio to the first cue for demo (could be improved with real mapping)
          if (mappedCues.length > 0) {
            mappedCues[0].audioBase64 = audioData.audio_base64;
            mappedCues[0].duration = audioData.duration_ms || mappedCues[0].duration;
          }
          setAudioCues([...mappedCues]);
        })
        .catch((err) => {
          // fallback to default mappedCues on audio-generation error
          setAudioCues(mappedCues);
        });

      setAudioCues(mappedCues);
    })
    .catch((err) => {
      // fallback to sample on error for demo
      setAudioCues(sampleAudioCues);
      // Optionally: set error state
      // setError(err.message);
    })
    .finally(() => setIsLoading(false));

  };

  const handleMasterMix = () => {
    setShowEvaluation(true);
  };

  return (
    <div className="min-h-screen bg-background relative">
      <FloatingParticles count={40} />
      
      <div className="relative z-10">
        {/* Header */}
        <motion.header 
          className="border-b border-border/30 backdrop-blur-sm bg-background/50 sticky top-0 z-50"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
        >
          <div className="container mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <motion.div 
                className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center"
                animate={{ 
                  boxShadow: [
                    "0 0 10px hsl(var(--primary) / 0.5)",
                    "0 0 20px hsl(var(--primary) / 0.8)",
                    "0 0 10px hsl(var(--primary) / 0.5)",
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <span className="text-primary-foreground font-display text-sm font-bold">BGM</span>
              </motion.div>
              <span className="font-display text-lg">
                <span className="text-foreground">Back</span>
                <span className="text-primary">Ground</span>
                <span className="text-foreground">Mellow</span>
              </span>
            </div>
            
            <div className="flex items-center gap-4">
              <span className="text-xs text-muted-foreground font-mono">v1.0.0-beta</span>
              <div className="w-2 h-2 rounded-full bg-secondary animate-pulse" />
            </div>
          </div>
        </motion.header>

        {/* Hero Section */}
        <HeroSection onDecompose={handleDecompose} />

        {/* Audio Cues Section */}
        <AnimatePresence>
          {audioCues.length > 0 && (
            <motion.section
              className="container mx-auto px-4 py-12"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -40 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center gap-2 mb-6">
                <div className="w-1 h-6 bg-gradient-to-b from-primary to-secondary rounded-full" />
                <h2 className="font-display text-xl tracking-wider text-foreground">
                  AUDIO CUES
                </h2>
                <span className="ml-2 px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-mono">
                  {audioCues.length}
                </span>
              </div>

              <div className="space-y-4">
                {audioCues.map((cue, index) => (
                  <motion.div
                    key={cue.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.15 }}
                  >
                    <AudioCard
                      id={cue.id}
                      type={cue.type}
                      prompt={cue.prompt}
                    />
                  </motion.div>
                ))}
              </div>

              {/* Master Mix Button */}
              <motion.div 
                className="mt-12 flex justify-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
              >
                <MasterMixButton 
                  onMix={handleMasterMix}
                  disabled={audioCues.length === 0}
                />
              </motion.div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Evaluation Section */}
        <AnimatePresence>
          {showEvaluation && (
            <motion.section
              className="container mx-auto px-4 pb-12"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <EvaluationGrid />
            </motion.section>
          )}
        </AnimatePresence>

        {/* Footer */}
        <footer className="border-t border-border/30 py-8 mt-12">
          <div className="container mx-auto px-4 text-center">
            <p className="text-xs text-muted-foreground">
              <span className="font-display tracking-wider">BackGroundMellow ENGINE</span>
              {" · "}
              <span className="font-mono">Powered by AI</span>
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Index;
