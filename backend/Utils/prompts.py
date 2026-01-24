from langchain_core.prompts import PromptTemplate

gemini_audio_prompt = PromptTemplate(
    input_variables=["story_text", "speed_wps"],
    template=(
        """Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

Story: "{story_text}"

Reading Speed: {speed_wps} words per second
Total Story Words: Count the words in the story
Total Duration (ms): Calculate as (total_words / {speed_wps}) * 1000

For each sound, you MUST provide:
- audio_class: detailed sound description for SoundGen AI
- audio_type: SFX (short sounds), AMBIENCE (background), or MUSIC (emotional)
- word_index: position (0-based) where the sound should start in the story
- start_time_ms: EXACT start time in milliseconds. Calculate as: (word_index / {speed_wps}) * 1000
- duration_ms: EXACT duration in milliseconds that YOU decide based on:
  * SFX: Decide duration (500-3000ms) based on the specific sound - a single bark might be 800ms, footsteps might be 2000ms, a door slam might be 1200ms
  * AMBIENCE: Decide duration based on story context - how long should this ambience play? Calculate from word_index to where it should end (next scene change, next AMBIENCE, or story end)
  * MUSIC: Decide duration based on emotional arc - how long should this musical element play? Consider the emotional moment and when it should fade (typically 2000-10000ms)
- weight_db: volume adjustment (-10.0 to 5.0, use 6.0 for "loud")

CRITICAL: You MUST provide a specific duration_ms value for EVERY audio cue. Do not leave it to be calculated later. Think about:
- For SFX: How long does this specific sound naturally last?
- For AMBIENCE: When does the scene/environment change in the story?
- For MUSIC: When does the emotional moment peak and fade?

Timing Calculation Rules:
1. start_time_ms = (word_index / {speed_wps}) * 1000 (round to nearest integer)
2. duration_ms = YOUR DECISION based on story context and sound type - provide the exact value
3. For overlapping sounds of the same type (e.g., two AMBIENCE cues), calculate when the first should end (typically when the second starts)
4. Ensure start_time_ms + duration_ms does not exceed total_duration_ms
5. Be precise - your duration_ms values will be used directly without modification

Try keeping as few Audio Cues as possible, not more than 3-4 Audio Cues.

Return ONLY a JSON array with these exact fields:
[
  {{"audio_class": "detailed sound description", "audio_type": "SFX|AMBIENCE|MUSIC", "word_index": 0, "start_time_ms": 0, "duration_ms": 2000, "weight_db": 0.0}}
]

"""
    ),
)
gemini_audio_prompt_with_narrator = PromptTemplate(
    input_variables=["story_text", "speed_wps"],
    template=(
        """
You are specialized agent good at analyzing stories and extracting audio sources (audio cues) with precise timing based on reading speed.
Analyze this story for cinematic sound design. Extract audio cues with precise timing based on reading speed.

Story: {story_text}
Reading Speed: {speed_wps} words per second
Total Story Words: Count the words in the story
Cinematic Master Model: Story Analysis & Sync Prompt
Role & Expertise: You are a Master Sound Designer and Narrative Director Agent. Your task is to perform a deep semantic analysis of the provided story to extract cinematic audio cues and direct a Narrator AI. You must ensure that the audio atmosphere perfectly syncs with the emotional arc and reading pace of the narrator.

### 1. NARRATOR AI DIRECTIVES The Narrator AI reads the entire story from word index 0 to the end. You must provide a "Narrator_Style" description that tells the AI exactly how to perform the reading based on the story's genre, mood, and tension.

Narrator Persona Guidelines: Match the story's context to one of these reference styles or create a custom blend:

Suspense/Horror: Low pitch, slower pace, breathless or whispering delivery.

Action/Urgency: Faster pace, higher intensity, clear and sharp articulation.

Serene/Nature: Calm, moderate pace, smooth intonation with subtle warmth.

Reference Examples for Narrator:
Aditi - Slightly High-Pitched, Expressive Tone: "Aditi speaks with a slightly higher pitch in a close-sounding environment. Her voice is clear, with subtle emotional depth and a normal pace, all captured in high-quality recording."

- Sita - Rapid, Slightly Monotone: "Sita speaks at a fast pace with a slightly low-pitched voice, captured clearly in a close-sounding environment with excellent recording quality."

- Tapan - Male, Moderate Pace, Slightly Monotone: "Tapan speaks at a moderate pace with a slightly monotone tone. The recording is clear, with a close sound and only minimal ambient noise."

- Sunita - High-Pitched, Happy Tone: "Sunita speaks with a high pitch in a close environment. Her voice is clear, with slight dynamic changes, and the recording is of excellent quality."

- Karan - High-Pitched, Positive Tone: "Karan's high-pitched, engaging voice is captured in a clear, close-sounding recording. His slightly slower delivery conveys a positive tone."

- Amrita - High-Pitched, Flat Tone: "Amrita speaks with a high pitch at a slow pace. Her voice is clear, with excellent recording quality and only moderate background noise."

- Aditi - Slow, Slightly Expressive: "Aditi speaks slowly with a high pitch and expressive tone. The recording is clear, showcasing her energetic and emotive voice."

- Young Male Speaker, American Accent: "A young male speaker with a high-pitched American accent delivers speech at a slightly fast pace in a clear, close-sounding recording."

- Bikram - High-Pitched, Urgent Tone: "Bikram speaks with a higher pitch and fast pace, conveying urgency. The recording is clear and intimate, with great emotional depth."

- Anjali - High-Pitched, Neutral Tone: "Anjali speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality."
Narrator Description Goal: Describe the emotion, pitch, and pace required for the specific story context.

### 2. AUDIO CUE ENGINEERING You must identify 3-4 critical audio cues that ground the story in a professional soundscape.

SFX (Short Effects): Punctuate specific actions (e.g., twig snapping, door slam). Duration: 500ms–3000ms.

AMBIENCE (Environment): Constant background textures (e.g., rain, forest hum). Duration: From the trigger word to the next scene change or end of story.

MUSIC (Emotional Score): Sets the heart of the scene (e.g., "Tense orchestral strings"). Duration: 2000ms–10000ms.

### 3. TIMING & SYNC MATH Precision is mandatory for "Multimodal Alignment".

word_index: The 0-based position of the word that triggers the sound.

start_time_ms: Calculated as (word_index / {speed_wps}) * 1000.

duration_ms: You must provide an exact value.

For Overlaps: If a new AMBIENCE starts, the previous one of the same type should end at that start_time_ms.

weight_db: volume adjustment (-15.0 to 6.0). 6.0 = "Defeaning/Loud".

### 4. OUTPUT CONSTRAINTS

JSON ONLY: No conversational filler.

Cue Count: Keep as less as possible.

Narrator Object: Include a single narrator_description at the root.

### EXAMPLE OF EXPECTED ANALYSIS Story: "The door creaked open. Rain lashed against the window as he stepped into the cold hall." (Speed: 2 wps)

JSON

{{

  "audio_cues": [
    {{
        "story": " The part of the story that the narrator will read with given descrpition , make sure to include pauses and breaks as per the narrator description
        
        # you might break story into multiple parts and make seprate audio cues for each part
        ", 
        "narrator_description": "Tapan speaks at a moderate pace with a low-pitched, gravelly tone to convey mystery. Clear, close-sounding recording with a cold, detached emotional depth",
        ##### audio que for narrator to know how to read the story
        "audio_type": "NARRATOR",
        "start_time_ms": 0,
        "duration_ms": duration_ms,
    }},
    {{
      "audio_class": "Heavy wooden door creaking open slowly with high-frequency friction",
      "audio_type": "SFX",
      "word_index": 1,
      "start_time_ms": 500,
      "duration_ms": 1500,
      "weight_db": 2.0
    }},
    {{
      "audio_class": "Heavy rain hitting glass window with distant thunder rumbles",
      "audio_type": "AMBIENCE",
      "word_index": 4,
      "start_time_ms": 2000,
      "duration_ms": 8000,
      "weight_db": -5.0
    }},
    {{
      "audio_class": "Dark cinematic suspense pad with low synth drones",
      "audio_type": "MUSIC",
      "word_index": 10,
      "start_time_ms": 5000,
      "duration_ms": 10000,
      "weight_db": 0.0
    }}
  ]
}}

"""
    ),
)
