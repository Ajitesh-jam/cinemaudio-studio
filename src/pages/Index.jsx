import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import HeroSection from "../components/HeroSection";
import AudioCard from "../components/AudioCard";
import MasterMixButton from "../components/MasterMixButton";
import FloatingParticles from "../components/FloatingParticles";
import AnimatedLoader from "../components/ui/AnimatedLoader";
import FinalAudioPlayer from "../components/FinalAudioPlayer";
import EvaluationForm from "../components/EvaluationForm.jsx";

// Google Drive and Sheets configuration
// TODO: Replace these with your actual Google Drive folder and Google Sheets links
const GOOGLE_DRIVE_FOLDER_LINK = import.meta.env.VITE_GOOGLE_DRIVE_FOLDER_LINK || "https://drive.google.com/drive/folders/12dhku5E1uZrFME68fRJd2rnEl_Le3y8P?usp=sharing";
const GOOGLE_SHEETS_LINK = import.meta.env.VITE_GOOGLE_SHEETS_LINK || "https://docs.google.com/spreadsheets/d/1EgWl5yfGMz0aOja6BAt8JiJnwo7MKCyHKQa3mqQSHWQ/edit?usp=sharing";

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
  const [storyText, setStoryText] = useState("");
  const [audioCues, setAudioCues] = useState([]);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [finalAudio, setFinalAudio] = useState(null);

  const handleDecompose = (storyText) => {

    // INSERT_YOUR_CODE
    // Use relative path to leverage Vite proxy in development
    const apiBase = import.meta.env.VITE_BACKEND_ENDPOINT || "/api";
    setIsLoading(true);
    fetch(`${apiBase}/v1/decide-cues`, {
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

      const cues = data?.cues || [];
      // For demonstration, wrap in default structure expected by AudioCard
      let mappedCues = cues.map((cue) => ({
        id: cue.id,
        audio_class: cue.audio_class || "",
        audio_type: cue.audio_type || "SFX",
        start_time_ms: cue.start_time_ms || 0,
        duration_ms: cue.duration_ms || 10,
        weight_db: cue.weight_db || 0,
        fade_ms: cue.fade_ms || 500,
        audioBase64: null
      }));

      console.log("mappedCues :", mappedCues);

      // INSERT_YOUR_CODE
      // Step 1: Call the /api/v1/generate-audio API on the backend
      fetch(`${apiBase}/v1/generate-audio`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          cues: mappedCues.map((cue) => ({
            id: cue.id,
            audio_class: cue.audio_class,
            audio_type: cue.audio_type,
            start_time_ms: cue.start_time_ms,
            duration_ms: cue.duration_ms,
            weight_db: cue.weight_db,
            fade_ms: cue.fade_ms
          })),
          total_duration_ms: data?.total_duration_ms ?? 1000
        })
      })
        .then(async (res) => {
          if (!res.ok) throw new Error("Failed to generate audio");
          return await res.json();
        })
        .then((audioData) => {
          // Update mappedCues with audio data
          if (mappedCues.length > 0) {
            for (let i = 0; i < mappedCues.length; i++) {
              if (
                audioData.audio_cues[i]?.audio_base64 !== null &&
                audioData.audio_cues[i]?.audio_base64 !== undefined &&
                audioData.audio_cues[i]?.audio_cue?.id === mappedCues[i].id
              ) {
                mappedCues[i].audioBase64 = audioData.audio_cues[i].audio_base64;
                mappedCues[i].duration_ms =
                  audioData.audio_cues[i].duration_ms || mappedCues[i].duration_ms;
              }
            }
          }
          
          // Filter out cues that don't have audio before setting state
          const cuesWithAudio = mappedCues.filter(cue => 
            cue.audioBase64 !== null && 
            cue.audioBase64 !== undefined && 
            cue.audioBase64 !== ""
          );
          
          console.log("mappedCues With Audio Base64:", cuesWithAudio);
          
          // Only set cues that have audio
          if (cuesWithAudio.length > 0) {
            setAudioCues(cuesWithAudio);
          } else {
            console.warn("No audio cues with audio data found");
            setAudioCues([]);
          }
          
          setIsLoading(false);
        })
        .catch((err) => {
          console.error("Error generating audio:", err);
          // Don't set cues without audio
          setAudioCues([]);
          setIsLoading(false);
        });
    })
    .catch((err) => {
      console.error("Error fetching audio cues:", err);
      setAudioCues([]);
      setIsLoading(false);
    });

  };

  const handleUpdate = (cueId, updates) => {
    setAudioCues(prevCues =>
      prevCues.map(cue =>
        cue.id === cueId ? { ...cue, ...updates } : cue
      )
    );
  };

  const handleMasterMix = () => {

    // Remove the extra duration_ms field from the root, since backend expects only audio_cue + audio_base64 per AudioCueWithAudioBase64
    const cues = audioCues.map((cue) => ({
      audio_cue: {
        id: cue.id,
        audio_class: cue.audio_class,
        audio_type: cue.audio_type,
        start_time_ms: cue.start_time_ms,
        duration_ms: cue.duration_ms,
        weight_db: cue.weight_db,
        fade_ms: cue.fade_ms
      },
      audio_base64: cue.audioBase64,
      duration_ms:cue.duration_ms
    }));
    console.log("cues :", cues);
    const apiBase = import.meta.env.VITE_BACKEND_ENDPOINT || "/api";
    console.log("storyText :", storyText);
    fetch(`${apiBase}/v1/generate-audio-cues-with-audio-base64`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        cues: cues,
        story_text: storyText,
        speed_wps: 2
      })
    })
    .then(async (response) => {
      if (!response.ok) throw new Error("Failed to generate audio");
      return response.json();
    })
    .then((data) => {
      console.log("data :", data);
      // Calculate duration from cues (max of start_time_ms + duration_ms)
      const totalDurationMs = Math.max(
        ...cues.map(cue => cue.audio_cue.start_time_ms + cue.audio_cue.duration_ms),
        0
      );
      const durationSeconds = totalDurationMs / 1000;
      
      // Store as object with audioBase64 and duration for FinalAudioPlayer
      setFinalAudio({
        audioBase64: data.audio_base64 || null,
        duration: durationSeconds
      });
    })
    .catch((err) => {
      console.error("Error generating audio:", err);
    })
    .finally(() => setIsLoading(false));
    setShowEvaluation(true);

    return true;
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
        <HeroSection isLoading={isLoading} onDecompose={handleDecompose} storyText={storyText} setStoryText={setStoryText} handleUpdate={handleUpdate} />

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
                {audioCues
                  .filter(cue => cue.audioBase64 && cue.audioBase64 !== null && cue.audioBase64 !== "")
                  .map((cue, index) => (
                    <motion.div
                      key={cue.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.15 }}
                    >
                      <AudioCard
                        id={cue.id}
                        type={cue.audio_type}
                        prompt={cue.audio_class}
                        start_time_ms={cue.start_time_ms}
                        duration_ms={cue.duration_ms}
                        weight_db={cue.weight_db}
                        fade_ms={cue.fade_ms}
                        audio_base64={cue.audioBase64}
                        handleUpdate={handleUpdate}
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
                  isLoading={isLoading}
                  onMix={(storyText) => handleMasterMix(storyText)}
                  disabled={audioCues.length === 0}
                />
              </motion.div>
            </motion.section>
          )}
        </AnimatePresence>


        {/* Final Audio & Evaluation */}
        <AnimatePresence>
          {finalAudio && finalAudio.audioBase64 && (
            <motion.section
              className="container mx-auto px-4 pb-12 space-y-6"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <FinalAudioPlayer 
                audioBase64={finalAudio.audioBase64}
                duration={finalAudio.duration}
              />
              <EvaluationForm 
                audioBase64={finalAudio.audioBase64}
                storyText={storyText}
                driveFolderLink={GOOGLE_DRIVE_FOLDER_LINK}
                sheetLink={GOOGLE_SHEETS_LINK}
              />
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
