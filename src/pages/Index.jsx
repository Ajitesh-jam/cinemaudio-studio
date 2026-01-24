import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import HeroSection from "../components/HeroSection";
import AudioCard from "../components/AudioCard";
import MasterMixButton from "../components/MasterMixButton";
import FloatingParticles from "../components/FloatingParticles";
import AnimatedLoader from "../components/ui/AnimatedLoader";
import FinalAudioPlayer from "../components/FinalAudioPlayer";
import EvaluationForm from "../components/EvaluationForm.jsx";


const Index = () => {
  const [storyText, setStoryText] = useState("");
  const [audioCues, setAudioCues] = useState([]);
  const [narratorCues, setNarratorCues] = useState([]);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [finalAudio, setFinalAudio] = useState(null);

  const handleDecompose = (storyText) => {

    //restore the audio cues
    setAudioCues([]);
    setNarratorCues([]);
    setFinalAudio(null);
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
        const cues = data?.cues || [];

        // Separate AudioCue and NarratorCue
        const audioCueList = [];
        const narratorCueList = [];

        cues.forEach((cue) => {
          const isNarrator = cue.audio_type === "NARRATOR" || cue.story || cue.narrator_description;

          if (isNarrator) {
            narratorCueList.push({
              id: cue.id,
              story: cue.story || "",
              narrator_description: cue.narrator_description || "",
              audio_type: "NARRATOR",
              start_time_ms: cue.start_time_ms || 0,
              duration_ms: cue.duration_ms || 10,
              audioBase64: null
            });
          } else {
            audioCueList.push({
              id: cue.id,
              audio_class: cue.audio_class || "",
              audio_type: cue.audio_type || "SFX",
              start_time_ms: cue.start_time_ms || 0,
              duration_ms: cue.duration_ms || 10,
              weight_db: cue.weight_db || 0,
              fade_ms: cue.fade_ms || 500,
              audioBase64: null
            });
          }
        });

        console.log("audioCueList:", audioCueList);
        console.log("narratorCueList:", narratorCueList);

        // Combine all cues for the generate-audio API call
        const allCues = [...audioCueList, ...narratorCueList];

        // INSERT_YOUR_CODE
        // Step 1: Call the /api/v1/generate-audio API on the backend
        fetch(`${apiBase}/v1/generate-audio`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            cues: allCues.map((cue) => {
              if (cue.audio_type === "NARRATOR") {
                return {
                  id: cue.id,
                  audio_type: cue.audio_type,
                  start_time_ms: cue.start_time_ms,
                  duration_ms: cue.duration_ms,
                  story: cue.story,
                  narrator_description: cue.narrator_description
                };
              } else {
                return {
                  id: cue.id,
                  audio_class: cue.audio_class,
                  audio_type: cue.audio_type,
                  start_time_ms: cue.start_time_ms,
                  duration_ms: cue.duration_ms,
                  weight_db: cue.weight_db,
                  fade_ms: cue.fade_ms
                };
              }
            }),
            total_duration_ms: data?.total_duration_ms ?? 1000
          })
        })
          .then(async (res) => {
            if (!res.ok) throw new Error("Failed to generate audio");
            return await res.json();
          })
          .then((audioData) => {
            // Update cues with audio data
            const updatedAudioCues = [...audioCueList];
            const updatedNarratorCues = [...narratorCueList];

            if (audioData.audio_cues && audioData.audio_cues.length > 0) {
              audioData.audio_cues.forEach((audioCueData) => {
                const cueId = audioCueData.audio_cue?.id;
                const audioBase64 = audioCueData.audio_base64;
                const durationMs = audioCueData.duration_ms;
                const cueType = audioCueData.audio_cue?.audio_type;

                if (audioBase64 && audioBase64 !== null && audioBase64 !== "") {
                  if (cueType === "NARRATOR") {
                    const narratorIndex = updatedNarratorCues.findIndex(c => c.id === cueId);
                    if (narratorIndex !== -1) {
                      updatedNarratorCues[narratorIndex].audioBase64 = audioBase64;
                      if (durationMs) {
                        updatedNarratorCues[narratorIndex].duration_ms = durationMs;
                      }
                    }
                  } else {
                    const audioIndex = updatedAudioCues.findIndex(c => c.id === cueId);
                    if (audioIndex !== -1) {
                      updatedAudioCues[audioIndex].audioBase64 = audioBase64;
                      if (durationMs) {
                        updatedAudioCues[audioIndex].duration_ms = durationMs;
                      }
                    }
                  }
                }
              });
            }

            // Filter out cues that don't have audio before setting state
            const cuesWithAudio = updatedAudioCues.filter(cue =>
              cue.audioBase64 !== null &&
              cue.audioBase64 !== undefined &&
              cue.audioBase64 !== ""
            );

            const narratorCuesWithAudio = updatedNarratorCues.filter(cue =>
              cue.audioBase64 !== null &&
              cue.audioBase64 !== undefined &&
              cue.audioBase64 !== ""
            );

            console.log("audioCues With Audio Base64:", cuesWithAudio);
            console.log("narratorCues With Audio Base64:", narratorCuesWithAudio);

            // Set both cue types
            setAudioCues(cuesWithAudio);
            setNarratorCues(narratorCuesWithAudio);

            if (cuesWithAudio.length === 0 && narratorCuesWithAudio.length === 0) {
              console.warn("No audio cues with audio data found");
            }

            setIsLoading(false);
          })
          .catch((err) => {
            console.error("Error generating audio:", err);
            // Don't set cues without audio
            setAudioCues([]);
            setNarratorCues([]);
            setIsLoading(false);
          });
      })
      .catch((err) => {
        console.error("Error fetching audio cues:", err);
        setAudioCues([]);
        setNarratorCues([]);
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

  const handleNarratorUpdate = (cueId, updates) => {
    setNarratorCues(prevCues =>
      prevCues.map(cue =>
        cue.id === cueId ? { ...cue, ...updates } : cue
      )
    );
  };

  const handleMasterMix = () => {

    // Combine audio cues and narrator cues for master mix
    const audioCuePayload = audioCues.map((cue) => ({
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
      duration_ms: cue.duration_ms
    }));

    const narratorCuePayload = narratorCues.map((cue) => ({
      audio_cue: {
        id: cue.id,
        story: cue.story,
        narrator_description: cue.narrator_description,
        audio_type: cue.audio_type,
        start_time_ms: cue.start_time_ms,
        duration_ms: cue.duration_ms
      },
      audio_base64: cue.audioBase64,
      duration_ms: cue.duration_ms
    }));

    const cues = [...audioCuePayload, ...narratorCuePayload];
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

        {/* Narrator Cues Section */}
        <AnimatePresence>
          {narratorCues.length > 0 && (
            <motion.section
              className="container mx-auto px-4 py-12"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -40 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center gap-2 mb-6">
                <div className="w-1 h-6 bg-gradient-to-b from-orange-500 to-orange-400 rounded-full" />
                <h2 className="font-display text-xl tracking-wider text-foreground">
                  NARRATOR CUES
                </h2>
                <span className="ml-2 px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-400 text-xs font-mono">
                  {narratorCues.length}
                </span>
              </div>

              <div className="space-y-4">
                {narratorCues
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
                        type="NARRATOR"
                        prompt={cue.story || "Narrator audio"}
                        start_time_ms={cue.start_time_ms}
                        duration_ms={cue.duration_ms}
                        weight_db={0}
                        fade_ms={500}
                        audio_base64={cue.audioBase64}
                        handleUpdate={handleNarratorUpdate}
                      />
                    </motion.div>
                  ))}
              </div>
            </motion.section>
          )}
        </AnimatePresence>

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
            </motion.section>
          )}
        </AnimatePresence>

        {/* Master Mix Button */}
        <AnimatePresence>
          {(audioCues.length > 0 || narratorCues.length > 0) && (
            <motion.section
              className="container mx-auto px-4 pb-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ delay: 0.5 }}
            >
              <div className="flex justify-center">
                <MasterMixButton
                  isLoading={isLoading}
                  onMix={(storyText) => handleMasterMix(storyText)}
                  disabled={audioCues.length === 0 && narratorCues.length === 0}
                />
              </div>
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
