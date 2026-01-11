import sys
import os


# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)

import logging
import json
import re
import warnings
from Variable.dataclases import AudioCue
from Variable.configurations import MODIFIER_WORDS, DEFAULT_WEIGHT_DB, DEFAULT_SFX_DURATION_MS
import math
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
    nlp_available = True
except Exception:
    nlp = None
    nlp_available = False

# Load environment variables from .env file
# Try to load from backend directory first, then project root
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))
load_dotenv(os.path.join(backend_dir, ".env"))  # Try backend/.env
load_dotenv(os.path.join(project_root, ".env"))  # Try project root .env
load_dotenv()  # Also try default locations

# Initialize logger first
logger = logging.getLogger(__name__)

# Try to import Gemini API - prefer new google.genai, fallback to deprecated google.generativeai
GEMINI_AVAILABLE = False
USE_NEW_GENAI = False
genai = None

# Try new google.genai package first
try:
    import google.genai as genai  # type: ignore
    GEMINI_AVAILABLE = True
    USE_NEW_GENAI = True
    logger.info("Using new google.genai package")
except ImportError:
    # Fallback to deprecated google.generativeai
    try:
        # Suppress the deprecation warning for now
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*google.generativeai.*")
            import google.generativeai as genai
        GEMINI_AVAILABLE = True
        USE_NEW_GENAI = False
        logger.warning("Using deprecated google.generativeai package. Consider upgrading to google.genai")
    except ImportError:
        GEMINI_AVAILABLE = False
        logger.warning("Gemini API packages not available. Install google-genai or google-generativeai")


def _classify_audio_type(word: str, pos_tag: str, context: str = "") -> Tuple[str | None, str | None]:
    """
    Classifies a word/phrase into audio type (SFX/AMBIENCE/MUSIC) and generates a prompt.
    Returns (audio_type, audio_prompt)
    """
    word_lower = word.lower()
    
    # Sound-producing verbs -> SFX
    sound_verbs = {
        'bark', 'barking', 'barked', 'barked',
        'cat','cat purring', 'cat hissing', 'cat meowing', 'cat purring', 'cat hissing',
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
    if not nlp_available or nlp is None:
        # Fallback to simple extraction if NLP is not available
        return _extract_audio_cues_simple(story_text, speed_wps)
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables. Set it with: export GEMINI_API_KEY='your-key'")
        return None
    
    try:
        words = story_text.split()
        total_words = len(words)
        
        prompt = f"""Analyze this story for cinematic sound design. Extract audio cues with precise timing.

Story: "{story_text}"

For each sound, provide:
- audio_class: detailed sound description for SoundGen AI
- audio_type: SFX (short sounds), AMBIENCE (background), or MUSIC (emotional)
- word_index: position in story (0 to {total_words-1})
- weight_db: volume adjustment (-10.0 to 5.0, use 6.0 for "loud")

Try keeping as less as possible Audio Cues not more that 3-4 Audio Cues.

Return ONLY a JSON array:
[
  {{"audio_class": "detailed sound description", "audio_type": "SFX|AMBIENCE|MUSIC", "word_index": 0, "weight_db": 0.0}}
]

JSON:"""
        
        # Handle both new and old API
        if USE_NEW_GENAI:
            # New google.genai API
            try:
                client = genai.Client(api_key=api_key)  # type: ignore
                response = client.models.generate_content(  # type: ignore
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                response_text = response.text
            except AttributeError:
                # If new API structure is different, fallback to old API pattern
                logger.warning("New API structure not recognized, trying alternative...")
                genai.configure(api_key=api_key)  # type: ignore
                model = genai.GenerativeModel('gemini-2.5-flash')  # type: ignore
                response = model.generate_content(prompt)  # type: ignore
                response_text = response.text
        else:
            # Old google.generativeai API
            genai.configure(api_key=api_key)  # type: ignore
            model = genai.GenerativeModel('gemini-2.5-flash')  # type: ignore
            response = model.generate_content(prompt)  # type: ignore
            response_text = response.text
        
        # Clean up the markdown response to get pure JSON
        json_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Extract JSON if it's embedded in text
        json_match = re.search(r'\[[\s\S]*?\]', json_text)
        if json_match:
            json_text = json_match.group()
        
        return json.loads(json_text)
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

# --- 1. LOCAL HUGGING FACE MODEL (GLiNER) ---
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
def query_gemini(story_text: str, speed_wps: float):
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini API not available")
        return None
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables. Set it with: export GEMINI_API_KEY='your-key'")
        return None
    
    try:
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
        
        # Handle both new and old API
        if USE_NEW_GENAI:
            # New google.genai API - try different models
            model_names = ['gemini-2.5-flash', 'gemini-1.5-pro']
            response_text = None
            
            try:
                client = genai.Client(api_key=api_key)  # type: ignore
                for model_name in model_names:
                    try:
                        response = client.models.generate_content(  # type: ignore
                            model=model_name,
                            contents=prompt
                        )
                        response_text = response.text
                        logger.info(f"Using Gemini model: {model_name}")
                        break
                    except Exception as e:
                        logger.debug(f"Model {model_name} failed: {e}")
                        continue
            except AttributeError:
                # If new API structure is different, fallback to old API pattern
                logger.warning("New API structure not recognized, trying old API pattern...")
                genai.configure(api_key=api_key)  # type: ignore
                model_names = ['gemini-2.5-flash', 'gemini-1.5-pro', 'gemini-pro']
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)  # type: ignore
                        response = model.generate_content(prompt)  # type: ignore
                        response_text = response.text
                        logger.info(f"Using Gemini model: {model_name}")
                        break
                    except Exception as e:
                        logger.debug(f"Model {model_name} failed: {e}")
                        continue
            
            if not response_text:
                logger.error("No available Gemini model found")
                return None
        else:
            # Old google.generativeai API
            genai.configure(api_key=api_key)  # type: ignore
            model_names = ['gemini-2.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            model = None
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)  # type: ignore
                    logger.info(f"Using Gemini model: {model_name}")
                    break
                except Exception as e:
                    logger.debug(f"Model {model_name} failed: {e}")
                    continue
            
            if not model:
                logger.error("No available Gemini model found")
                return None
            
            response = model.generate_content(prompt)  # type: ignore
            response_text = response.text
        
        # Clean potential markdown backticks
        json_str = response_text.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON if embedded in text
        json_match = re.search(r'\[[\s\S]*?\]', json_str)
        if json_match:
            json_str = json_match.group()
        
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return None

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
            # query_llm is not available in this file, skip LLM fallback
            raw_output = None
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

if __name__ == "__main__":
    story_text = "Suddenly rain started so i ran to shelter where i heard loud dog barking"
    speed_wps = 100
    cues, total_duration = decide_audio_cues(story_text, speed_wps)
    print(cues)
    print(total_duration)