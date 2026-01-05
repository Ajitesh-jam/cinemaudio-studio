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
  },
  {
    id: 2,
    type: "SFX",
    prompt: "Car engine starting, tires on gravel",
  },
  {
    id: 3,
    type: "Music",
    prompt: "Tense orchestral underscore, low strings",
  },
];

const Index = () => {
  const [audioCues, setAudioCues] = useState([]);
  const [showEvaluation, setShowEvaluation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleDecompose = (storyText) => {
    setAudioCues(sampleAudioCues);
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
                <span className="text-primary-foreground font-display text-sm font-bold">CA</span>
              </motion.div>
              <span className="font-display text-lg">
                <span className="text-foreground">Cinem</span>
                <span className="text-primary">Audio</span>
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
              <span className="font-display tracking-wider">CINEMAUDIO ENGINE</span>
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
