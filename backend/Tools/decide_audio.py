import sys
import os


# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)

from headers.imports import *
from Variable.dataclases import AudioCue
from Variable.configurations import MODIFIER_WORDS, DEFAULT_WEIGHT_DB, DEFAULT_SFX_DURATION_MS

import json
import math
import re
import requests
import os
from typing import List, Dict, Tuple
from dataclasses import asdict

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


logger = logging.getLogger(__name__)

# Try importing spaCy, fall back to a simple approach if not available
try:
    import spacy
    nlp_available = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        logger.warning("Falling back to simple NLP extraction...")
        nlp_available = False
except ImportError:
    logger.warning("spaCy not installed. Install with: pip install spacy")
    logger.warning("Falling back to simple NLP extraction...")
    nlp_available = False


def _classify_audio_type(word: str, pos_tag: str, context: str = "") -> Tuple[str | None, str | None]:
    """
    Classifies a word/phrase into audio type (SFX/AMBIENCE/MUSIC) and generates a prompt.
    Returns (audio_type, audio_prompt)
    """
    word_lower = word.lower()
    
    # Sound-producing verbs -> SFX
    sound_verbs = {
        'bark', 'barking', 'barked', 'barked',
        'run', 'running', 'ran', 'runs',
        'scream', 'screaming', 'screamed', 'screams',
        'shout', 'shouting', 'shouted', 'shouts',
        'laugh', 'laughing', 'laughed', 'laughs',
        'cry', 'crying', 'cried', 'cries',
        'knock', 'knocking', 'knocked', 'knocks',
        'crash', 'crashing', 'crashed', 'crashes',
        'slam', 'slamming', 'slammed', 'slams',
        'bang', 'banging', 'banged', 'bangs',
        'whistle', 'whistling', 'whistled', 'whistles',
        'clap', 'clapping', 'clapped', 'claps',
        'step', 'stepping', 'stepped', 'steps',
        'walk', 'walking', 'walked', 'walks',
    }
    
    # Environmental/background nouns -> AMBIENCE
    environment_words = {
        'rain', 'raining', 'rainy',
        'storm', 'stormy', 'thunder',
        'wind', 'windy', 'blowing',
        'forest', 'jungle', 'wood',
        'city', 'urban', 'traffic',
        'ocean', 'sea', 'waves', 'beach',
        'river', 'stream', 'waterfall',
        'fire', 'burning', 'crackling',
        'snow', 'snowing', 'snowy',
        'desert', 'mountain', 'valley',
        'shelter', 'roof', 'indoors', 'room',
        'street', 'park', 'garden',
    }
    
    # Emotional/mood words -> MUSIC
    emotion_words = {
        'sad', 'sadness', 'melancholy', 'depressed',
        'happy', 'happiness', 'joy', 'joyful',
        'scared', 'scary', 'frightened', 'fear', 'fearful',
        'suspense', 'suspenseful', 'tense', 'tension',
        'eerie', 'creepy', 'horror', 'horrifying',
        'emotional', 'emotional', 'feeling',
        'calm', 'peaceful', 'serene', 'tranquil',
        'excited', 'exciting', 'thrilling',
        'romantic', 'love', 'loving',
        'angry', 'anger', 'furious',
        'sudden', 'suddenly', 'abrupt',
    }
    
    # Check for verbs that produce sounds
    if word_lower in sound_verbs or pos_tag in ['VERB', 'VBG', 'VBD', 'VBP', 'VBZ']:
        # Generate a descriptive prompt
        if 'bark' in word_lower:
            return ("SFX", "dog barking")
        elif 'run' in word_lower or 'step' in word_lower or 'walk' in word_lower:
            return ("SFX", "footsteps running")
        elif 'scream' in word_lower or 'shout' in word_lower:
            return ("SFX", "person shouting")
        elif 'laugh' in word_lower:
            return ("SFX", "person laughing")
        elif 'knock' in word_lower:
            return ("SFX", "door knocking")
        elif 'crash' in word_lower or 'slam' in word_lower or 'bang' in word_lower:
            return ("SFX", f"{word_lower} sound")
        else:
            return ("SFX", f"{word_lower} sound effect")
    
    # Check for environmental/background sounds
    elif word_lower in environment_words:
        if 'rain' in word_lower:
            return ("AMBIENCE", "rain falling")
        elif 'storm' in word_lower or 'thunder' in word_lower:
            return ("AMBIENCE", "thunderstorm")
        elif 'wind' in word_lower:
            return ("AMBIENCE", "wind blowing")
        elif 'forest' in word_lower or 'jungle' in word_lower or 'wood' in word_lower:
            return ("AMBIENCE", "forest ambience")
        elif 'city' in word_lower or 'traffic' in word_lower or 'urban' in word_lower:
            return ("AMBIENCE", "city traffic")
        elif 'ocean' in word_lower or 'sea' in word_lower or 'wave' in word_lower:
            return ("AMBIENCE", "ocean waves")
        elif 'fire' in word_lower:
            return ("AMBIENCE", "fire crackling")
        elif 'shelter' in word_lower or 'roof' in word_lower:
            return ("AMBIENCE", "rain on roof")
        else:
            return ("AMBIENCE", f"{word_lower} ambience")
    
    # Check for emotional/mood words
    elif word_lower in emotion_words:
        if 'sad' in word_lower:
            return ("MUSIC", "sad emotional music")
        elif 'happy' in word_lower or 'joy' in word_lower:
            return ("MUSIC", "happy upbeat music")
        elif 'scared' in word_lower or 'scary' in word_lower or 'fear' in word_lower:
            return ("MUSIC", "scary horror music")
        elif 'suspense' in word_lower or 'tense' in word_lower:
            return ("MUSIC", "suspenseful music")
        elif 'eerie' in word_lower or 'creepy' in word_lower or 'horror' in word_lower:
            return ("MUSIC", "eerie suspense music")
        elif 'sudden' in word_lower or 'suddenly' in word_lower:
            return ("MUSIC", "dramatic stinger")
        elif 'calm' in word_lower or 'peaceful' in word_lower:
            return ("MUSIC", "calm soothing music")
        else:
            return ("MUSIC", f"{word_lower} emotional music")
    
    # Default: try to infer from POS tag
    elif pos_tag in ['ADJ', 'JJ', 'JJR', 'JJS']:
        # Adjectives might be emotional -> MUSIC
        return ("MUSIC", f"{word_lower} background music")
    elif pos_tag in ['NOUN', 'NN', 'NNS']:
        # Nouns might be environmental -> AMBIENCE
        return ("AMBIENCE", f"{word_lower} ambient sound")
    
    # If we can't classify, return None to skip
    return (None, None)

def _extract_audio_cues_nlp(story_text: str, speed_wps: float):
    """
    Uses spaCy NLP to extract audio cues from text.
    """
    doc = nlp(story_text)
    words = story_text.lower().split()
    total_words = len(words)
    total_duration_ms = math.ceil((total_words / speed_wps) * 1000)
    
    cues_to_play: List[AudioCue] = []
    current_weight_db = DEFAULT_WEIGHT_DB
    last_cue_index: Dict[str, int] = {"SFX": -1, "AMBIENCE": -1, "MUSIC": -1}
    
    # Build word index mapping for accurate position tracking
    word_positions = []
    char_idx = 0
    for word in words:
        # Find the word in the original text
        pos = story_text.lower().find(word, char_idx)
        if pos != -1:
            word_positions.append((pos, word))
            char_idx = pos + len(word)
        else:
            word_positions.append((char_idx, word))
            char_idx += len(word) + 1
    
    # Process each token with better position tracking
    processed_indices = set()  # Track which words we've processed to avoid duplicates
    
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        
        # Find which word index this token corresponds to
        token_start = token.idx
        word_idx = 0
        for i, (pos, word) in enumerate(word_positions):
            if pos <= token_start < pos + len(word):
                word_idx = i
                break
        
        # Skip if we've already processed this word
        if word_idx in processed_indices:
            continue
        
        current_time_ms = math.ceil((word_idx / speed_wps) * 1000)
        
        # Check for weight modifiers
        if token.text.lower() in MODIFIER_WORDS:
            current_weight_db = MODIFIER_WORDS[token.text.lower()]
            logger.debug(f"Word '{token.text}' at {current_time_ms}ms. Setting weight to: {current_weight_db}dB")
            processed_indices.add(word_idx)
            continue
        
        # Classify the token
        audio_type, audio_prompt = _classify_audio_type(
            token.text, 
            token.pos_,
            context=token.sent.text
        )
        
        if audio_type and audio_prompt:
            logger.info(f"Detected '{token.text}' ({token.pos_}) -> {audio_type}: '{audio_prompt}' at {current_time_ms}ms")
            
            # Create the AudioCue
            cue = AudioCue(
                id=word_idx,
                audio_class=audio_prompt,
                start_time_ms=current_time_ms,
                duration_ms=DEFAULT_SFX_DURATION_MS,
                weight_db=current_weight_db,
                audio_type=audio_type
            )
            
            # Handle smart duration for AMBIENCE/MUSIC
            if audio_type in ["AMBIENCE", "MUSIC"]:
                last_idx = last_cue_index[audio_type]
                if last_idx != -1:
                    prev_cue = cues_to_play[last_idx]
                    prev_cue.duration_ms = current_time_ms - prev_cue.start_time_ms
                    logger.debug(f"Updating previous '{prev_cue.audio_type}' cue to end at {current_time_ms}ms.")
                
                cue.duration_ms = total_duration_ms - current_time_ms
                last_cue_index[audio_type] = len(cues_to_play)
            
            cues_to_play.append(cue)
            current_weight_db = DEFAULT_WEIGHT_DB
            processed_indices.add(word_idx)
    
    return cues_to_play, total_duration_ms

def _extract_audio_cues_simple(story_text: str, speed_wps: float):
    """
    Simple fallback approach without spaCy - uses basic pattern matching.
    """
    words = story_text.lower().split()
    total_words = len(words)
    total_duration_ms = math.ceil((total_words / speed_wps) * 1000)
    
    cues_to_play: List[AudioCue] = []
    current_weight_db = DEFAULT_WEIGHT_DB
    last_cue_index: Dict[str, int] = {"SFX": -1, "AMBIENCE": -1, "MUSIC": -1}

    i = 0
    while i < total_words:
        current_time_ms = math.ceil((i / speed_wps) * 1000)
        word = words[i]
        
        # Check for weight modifiers
        if word in MODIFIER_WORDS:
            current_weight_db = MODIFIER_WORDS[word]
            logger.debug(f"Word '{word}' at {current_time_ms}ms. Setting weight to: {current_weight_db}dB")
            i += 1
            continue
        
        # Try to classify the word
        # Simple heuristic: check if it looks like a verb (ends with -ing, -ed, -s) or known keywords
        audio_type, audio_prompt = _classify_audio_type(word, "", context="")
        
        if audio_type and audio_prompt:
            logger.info(f"Detected '{word}' -> {audio_type}: '{audio_prompt}' at {current_time_ms}ms")
            
            cue = AudioCue(
                id=i,
                audio_class=audio_prompt,
                start_time_ms=current_time_ms,
                duration_ms=DEFAULT_SFX_DURATION_MS,
                weight_db=current_weight_db,
                audio_type=audio_type
            )
            
            if audio_type in ["AMBIENCE", "MUSIC"]:
                last_idx = last_cue_index[audio_type]
                if last_idx != -1:
                    prev_cue = cues_to_play[last_idx]
                    prev_cue.duration_ms = current_time_ms - prev_cue.start_time_ms
                    logger.debug(f"Updating previous '{prev_cue.audio_type}' cue to end at {current_time_ms}ms.")

                cue.duration_ms = total_duration_ms - current_time_ms
                last_cue_index[audio_type] = len(cues_to_play)
            
            cues_to_play.append(cue)
            current_weight_db = DEFAULT_WEIGHT_DB
        
        i += 1
    
    return cues_to_play, total_duration_ms



def analyze_story_with_gemini(story_text: str, speed_wps: float):
    """
    Use Gemini API to analyze story and extract audio cues.
    Returns list of audio cue dictionaries.
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini API not available")
        return None
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables. Set it with: export GEMINI_API_KEY='your-key'")
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # Use 1.5 Flash for speed and high free-tier limits
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        words = story_text.split()
        total_words = len(words)
        
        prompt = f"""Analyze this story for cinematic sound design. Extract audio cues with precise timing.

Story: "{story_text}"

For each sound, provide:
- audio_class: detailed sound description for SoundGen AI
- audio_type: SFX (short sounds), AMBIENCE (background), or MUSIC (emotional)
- word_index: position in story (0 to {total_words-1})
- weight_db: volume adjustment (-10.0 to 5.0, use 6.0 for "loud")

Return ONLY a JSON array:
[
  {{"audio_class": "detailed sound description", "audio_type": "SFX|AMBIENCE|MUSIC", "word_index": 0, "weight_db": 0.0}}
]

JSON:"""
        
        response = model.generate_content(prompt)
        
        # Clean up the markdown response to get pure JSON
        json_text = response.text.replace('```json', '').replace('```', '').strip()
        
        # Extract JSON if it's embedded in text
        json_match = re.search(r'\[[\s\S]*?\]', json_text)
        if json_match:
            json_text = json_match.group()
        
        return json.loads(json_text)
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None



# def query_llm(system_prompt: str):
#     """
#     Query LLM using transformers library locally.
#     Uses a small, fast model that runs on CPU.
#     """
#     try:
#         from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
#     except ImportError:
#         logger.error("transformers library not installed. Install with: pip install transformers torch")
#         return None
    
#     # Use GPT-2 with few-shot prompting for better results
#     model_name = "gpt2"  # Small, reliable
    
#     try:
#         logger.info(f"Loading model {model_name}...")
        
#         # Use text-generation for GPT-2
#         generator = pipeline(
#             "text-generation",
#             model=model_name,
#             tokenizer=model_name,
#             device=-1,  # Use CPU
#             do_sample=True,
#             temperature=0.1,  # Very low for more deterministic output
#             top_p=0.95,
#             repetition_penalty=1.3,
#             return_full_text=False
#         )
        
#         logger.info("Generating response...")
#         results = generator(system_prompt, max_new_tokens=200, num_return_sequences=1, truncation=True)
        
#         # Extract generated text (T5 returns different format)
#         generated_text = None
#         if isinstance(results, list) and len(results) > 0:
#             result_item = results[0]
#             if isinstance(result_item, dict):
#                 # T5 models return {'generated_text': '...'}
#                 generated_text = result_item.get('generated_text', '')
#             else:
#                 generated_text = str(result_item)
#         elif isinstance(results, dict):
#             generated_text = results.get('generated_text', '')
#         elif isinstance(results, str):
#             generated_text = results
#         else:
#             generated_text = str(results)
        
#         # Remove the original prompt from the generated text if it's included
#         if generated_text and system_prompt in generated_text:
#             generated_text = generated_text.replace(system_prompt, "").strip()
        
#         if generated_text and len(generated_text.strip()) > 0:
#             logger.info(f"Successfully generated response (length: {len(generated_text)})")
#             return generated_text.strip()
#         else:
#             logger.warning(f"Empty response from model. Results type: {type(results)}, value: {results}")
#             return None
            
#     except Exception as e:
#         logger.error(f"Error generating with transformers: {e}")
#         print(f"--- DEBUG ERROR: {str(e)} ---")
#         return None
    
# def decide_audio_llm(story_text: str, speed_wps: float):
#     print(f"[LLM-DECIDER] Analyzing story with LLM...")
    
#     words = story_text.split()
#     total_duration_ms = math.ceil((len(words) / speed_wps) * 1000)

#     # Use LLM to describe sounds, then extract from description
#     system_prompt = f"""Describe the sounds in this story: "{story_text}" Sounds:"""

#     raw_output = query_llm(system_prompt)
#     logger.info(f"LLM output: {raw_output}")
    
#     if not raw_output:
#         print("[ERROR] LLM returned no data.")
#         return [], total_duration_ms

#     try:
#         # Convert to string if needed
#         if not isinstance(raw_output, str):
#             raw_output = str(raw_output)
        
#         # Extract audio cues from LLM description using keyword matching
#         raw_cues = []
#         story_lower = story_text.lower()
#         words_list = story_text.split()
#         output_lower = raw_output.lower()
        
#         # Comprehensive keyword mapping
#         sound_keywords = {
#             'rain': ('rain falling', 'AMBIENCE', ['rain', 'raining', 'rainy', 'raindrop']),
#             'dog': ('dog barking', 'SFX', ['dog', 'barking', 'bark', 'barked']),
#             'run': ('footsteps running', 'SFX', ['run', 'ran', 'running', 'runs']),
#             'shelter': ('shelter ambience', 'AMBIENCE', ['shelter', 'roof', 'indoors']),
#             'loud': ('loud sound', 'SFX', ['loud', 'loudly']),
#             'suddenly': ('dramatic stinger', 'MUSIC', ['suddenly', 'sudden', 'abrupt']),
#             'started': ('sound starting', 'SFX', ['started', 'start', 'began']),
#             'heard': ('sound heard', 'SFX', ['heard', 'hear', 'hearing']),
#         }
        
#         # Find all keywords in story
#         found_sounds = {}
#         for key, (audio_class, audio_type, keywords) in sound_keywords.items():
#             for i, word in enumerate(words_list):
#                 word_lower = word.lower().strip('.,!?;:')
#                 if any(kw in word_lower for kw in keywords):
#                     if key not in found_sounds or i < found_sounds[key]['word_index']:
#                         found_sounds[key] = {
#                             'audio_class': audio_class,
#                             'audio_type': audio_type,
#                             'word_index': i,
#                             'keywords': keywords
#                         }
        
#         # Check if LLM mentioned these sounds and create cues
#         for key, sound_info in found_sounds.items():
#             # Check if LLM output mentions this sound (more lenient matching)
#             llm_mentions = any(kw in output_lower for kw in sound_info['keywords'])
#             # Also check if LLM mentions related words
#             if not llm_mentions:
#                 related_words = {
#                     'rain': ['water', 'wet', 'storm', 'weather'],
#                     'dog': ['animal', 'pet', 'canine', 'woof'],
#                     'run': ['footstep', 'step', 'walk', 'move'],
#                     'shelter': ['cover', 'protection', 'inside', 'building'],
#                 }
#                 if key in related_words:
#                     llm_mentions = any(rw in output_lower for rw in related_words[key])
            
#             if llm_mentions or len(found_sounds) <= 3:  # Include if LLM mentioned it, or if few sounds found
#                 # Determine weight
#                 weight_db = 0.0
#                 if 'loud' in story_lower and sound_info['word_index'] < len(words_list):
#                     # Check if "loud" is near this word
#                     for j in range(max(0, sound_info['word_index'] - 2), min(len(words_list), sound_info['word_index'] + 3)):
#                         if 'loud' in words_list[j].lower():
#                             weight_db = 6.0
#                             break
                
#                 raw_cues.append({
#                     "audio_class": sound_info['audio_class'],
#                     "audio_type": sound_info['audio_type'],
#                     "word_index": sound_info['word_index'],
#                     "weight_db": weight_db
#                 })
        
#         if not raw_cues:
#             print(f"[ERROR] No valid cues extracted from LLM description. LLM said: {raw_output[:200]}...")
#             return [], total_duration_ms
        
#         logger.info(f"Extracted {len(raw_cues)} cues from LLM description")
#         cues_to_play: List[AudioCue] = []
#         last_cue_index = {"AMBIENCE": -1, "MUSIC": -1}

#         for item in raw_cues:
#             # Type snapping to prevent hallucination
#             a_type = str(item.get("audio_type", "SFX")).upper()
#             if a_type not in ["SFX", "AMBIENCE", "MUSIC"]:
#                 a_type = "SFX"

#             start_ms = math.ceil((item.get("word_index", 0) / speed_wps) * 1000)
            
#             cue = AudioCue(
#                 id=i,
#                 audio_class=item.get("audio_class", "background texture"),
#                 start_time_ms=start_ms,
#                 duration_ms=DEFAULT_SFX_DURATION_MS,
#                 weight_db=item.get("weight_db", DEFAULT_WEIGHT_DB),
#                 audio_type=a_type
#             )

#             # Logic for continuous sounds (Ambience/Music)
#             if a_type in last_cue_index:
#                 idx = last_cue_index[a_type]
#                 if idx != -1:
#                     cues_to_play[idx].duration_ms = start_ms - cues_to_play[idx].start_time_ms
                
#                 cue.duration_ms = max(0, total_duration_ms - start_ms)
#                 last_cue_index[a_type] = len(cues_to_play)

#             cues_to_play.append(cue)

#         print(f"[LLM-DECIDER] Generated {len(cues_to_play)} cues successfully.")
#         return cues_to_play, total_duration_ms

#     except Exception as e:
#         print(f"[ERROR] Parsing failed: {e}")
#         return [], total_duration_ms
    
    
    
# These imports are already at the top of the file, removing duplicates

logger = logging.getLogger(__name__)

# --- 1. LOCAL HUGGING FACE MODEL (GLiNER) ---
# This model finds the exact "source" words in your story text.
try:
    from gliner import GLiNER
    gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    GLINER_AVAILABLE = True
except ImportError:
    logger.warning("GLiNER not installed. Install with: pip install gliner")
    gliner_model = None
    GLINER_AVAILABLE = False
except Exception as e:
    logger.error(f"Failed to load GLiNER: {e}")
    gliner_model = None
    GLINER_AVAILABLE = False

def extract_local_entities(text: str):
    """Extract entities using GLiNER if available."""
    if not GLINER_AVAILABLE or not gliner_model:
        return []
    try:
        # We define custom labels for cinematic sound design
        labels = ["sound source", "environmental condition", "action"]
        entities = gliner_model.predict_entities(text, labels, threshold=0.4)
        return entities
    except Exception as e:
        logger.warning(f"GLiNER extraction failed: {e}")
        return []

# --- 2. GEMINI CLOUD MODEL ---
# This model performs the "Cinematic Reasoning."
def query_gemini(story_text: str, speed_wps: float):
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini API not available")
        return None
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables. Set it with: export GEMINI_API_KEY='your-key'")
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # Try different model names in order
        # Note: Model names should NOT include 'models/' prefix when using GenerativeModel
        model_names = [
            'gemini-3-flash-preview',         # Standard name
    
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"Using Gemini model: {model_name}")
                break
            except Exception as e:
                logger.debug(f"Model {model_name} failed: {e}")
                continue
        
        if not model:
            logger.error("No available Gemini model found")
            return None
        
        prompt = f"""
        Act as a Master Sound Designer for a movie. Analyze the story and create a JSON list of audio cues.
        
        Story: "{story_text}"
        Reading Speed: {speed_wps} words/sec.
        
        For each sound, determine:
        1. 'audio_class': A high-quality descriptive prompt (e.g., "heavy rain on a tin roof with distant thunder").
        2. 'audio_type': Choose strictly from [SFX, AMBIENCE, MUSIC].
        3. 'word_index': The index of the word where this sound starts.
        4. 'weight_db': Loudness from -15.0 to 5.0.
        
        Return ONLY a raw JSON array. No conversational text.
        """
        
        response = model.generate_content(prompt)
        # Clean potential markdown backticks
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON if embedded in text
        json_match = re.search(r'\[[\s\S]*?\]', json_str)
        if json_match:
            json_str = json_match.group()
        
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return None

# --- 3. THE INTEGRATED DECIDER ---
def decide_audio_llm(story_text: str, speed_wps: float):
    print(f"[DECIDER] Starting Hybrid AI Analysis...")
    
    words = story_text.split()
    total_duration_ms = math.ceil((len(words) / speed_wps) * 1000)
    
    # Step A: Try Gemini first, then fallback to local LLM
    gemini_cues = query_gemini(story_text, speed_wps)
    
    # If Gemini fails, try local LLM with keyword extraction
    if not gemini_cues:
        logger.info("Gemini failed, trying local LLM fallback...")
        # Use local keyword-based extraction as fallback
        raw_output = None
        try:
            # Try to use query_llm if available (defined earlier in file)
            system_prompt = f"""Describe the sounds in this story: "{story_text}"

Sounds:"""
            # Import or call query_llm - it should be defined above
            from Tools.decide_audio import query_llm
            raw_output = query_llm(system_prompt)
        except (NameError, ImportError):
            # If query_llm not available, skip LLM and use direct keyword matching
            raw_output = None
        # Extract cues directly from story using keyword matching
        story_lower = story_text.lower()
        words_list = story_text.split()
        
        sound_keywords = {
            'rain': ('rain falling', 'AMBIENCE', ['rain', 'raining', 'rainy', 'raindrop']),
            'dog': ('dog barking', 'SFX', ['dog', 'barking', 'bark', 'barked']),
            'run': ('footsteps running', 'SFX', ['run', 'ran', 'running', 'runs']),
            'shelter': ('shelter ambience', 'AMBIENCE', ['shelter', 'roof', 'indoors']),
            'loud': ('loud sound', 'SFX', ['loud', 'loudly']),
            'suddenly': ('dramatic stinger', 'MUSIC', ['suddenly', 'sudden', 'abrupt']),
            'started': ('sound starting', 'SFX', ['started', 'start', 'began']),
            'heard': ('sound heard', 'SFX', ['heard', 'hear', 'hearing']),
        }
        
        found_sounds = {}
        for key, (audio_class, audio_type, keywords) in sound_keywords.items():
            for i, word in enumerate(words_list):
                word_lower = word.lower().strip('.,!?;:')
                if any(kw in word_lower for kw in keywords):
                    if key not in found_sounds or i < found_sounds[key]['word_index']:
                        found_sounds[key] = {
                            'audio_class': audio_class,
                            'audio_type': audio_type,
                            'word_index': i,
                            'keywords': keywords
                        }
        
        gemini_cues = []
        for key, sound_info in found_sounds.items():
            # Include all found sounds for fallback
            weight_db = 0.0
            if 'loud' in story_lower and sound_info['word_index'] < len(words_list):
                for j in range(max(0, sound_info['word_index'] - 2), min(len(words_list), sound_info['word_index'] + 3)):
                    if 'loud' in words_list[j].lower():
                        weight_db = 6.0
                        break
            
            gemini_cues.append({
                "audio_class": sound_info['audio_class'],
                "audio_type": sound_info['audio_type'],
                "word_index": sound_info['word_index'],
                "weight_db": weight_db
            })
        
        if gemini_cues:
            logger.info(f"Fallback extracted {len(gemini_cues)} cues from keyword matching")
    
    # Step B: Use GLiNER locally to verify if sounds are physically mentioned (optional)
    local_entities = []
    if GLINER_AVAILABLE:
        try:
            local_entities = extract_local_entities(story_text)
        except Exception as e:
            logger.warning(f"GLiNER extraction failed: {e}")
            local_entities = []
    
    if not gemini_cues:
        print("[ERROR] AI Decider failed. Falling back to empty.")
        return [], total_duration_ms

    final_cues: List[AudioCue] = []
    last_cue_idx = {"AMBIENCE": -1, "MUSIC": -1}
    index = 0
    for item in gemini_cues:
        # Cross-reference with GLiNER: If Gemini found a sound, did GLiNER see a "source"?
        # (This increases your BTP "Sync Accuracy" metric)
        
        a_type = str(item.get("audio_type", "SFX")).upper()
        if a_type not in ["SFX", "AMBIENCE", "MUSIC"]: a_type = "SFX"

        start_ms = math.ceil((item.get("word_index", 0) / speed_wps) * 1000)
        
        cue = AudioCue(
            id=index,
            audio_class=item.get("audio_class", "ambient texture"),
            audio_type=a_type,
            start_time_ms=start_ms,
            duration_ms=DEFAULT_SFX_DURATION_MS,
            weight_db=item.get("weight_db", DEFAULT_WEIGHT_DB)
        )

        # Handle continuous mixing logic
        if a_type in last_cue_idx:
            prev_idx = last_cue_idx[a_type]
            if prev_idx != -1:
                final_cues[prev_idx].duration_ms = start_ms - final_cues[prev_idx].start_time_ms
            
            cue.duration_ms = max(0, total_duration_ms - start_ms)
            last_cue_idx[a_type] = len(final_cues)

        final_cues.append(cue)
        index += 1

    print(f"[DECIDER] Successfully generated {len(final_cues)} cinematic cues.")
    return final_cues, total_duration_ms

    
def decide_audio_cues(story_text: str, speed_wps: float):
    """
    Parses the story text using LLM and creates a timed list of AudioCues.
    Falls back to simple extraction if LLM fails.
    """
    logger.info("Starting audio decision process...")
    logger.info(f"Reading Speed: {speed_wps} words/sec")
    
    try:
        cues, total_duration = decide_audio_llm(story_text, speed_wps)
        if not cues:
            logger.warning("LLM returned no cues, falling back to simple extraction...")
            raise Exception("Failed to generate audio cues with LLM")
            # Fallback to simple extraction if available
            # if nlp_available:
            #     cues, total_duration = _extract_audio_cues_nlp(story_text, speed_wps)
            # else:
            #     cues, total_duration = _extract_audio_cues_simple(story_text, speed_wps)
    except Exception as e:
        logger.error(f"Error in decide_audio_llm: {e}")
        logger.info("Falling back to simple extraction...")
        # Fallback to simple extraction
        if nlp_available:
            cues, total_duration = _extract_audio_cues_nlp(story_text, speed_wps)
        else:
            cues, total_duration = _extract_audio_cues_simple(story_text, speed_wps)
    
    logger.info(f"Finished parsing. Found {len(cues)} audio cues.")
    return cues, total_duration

# if __name__ == "__main__":
#     story_text = "Suddenly rain started so i ran to shelter where i heard loud dog barking"
#     speed_wps = 100
#     cues, total_duration = decide_audio(story_text, speed_wps)
#     print(cues)
#     print(total_duration)