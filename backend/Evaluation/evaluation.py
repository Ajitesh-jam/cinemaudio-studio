"""
Evaluation runner for the audio-generation pipeline.

This script:
1) Reads a JSONL dataset (e.g. `yt_random_test_dataset.jsonl`)
2) Runs cue decision → specialist generation → optional missing fill → superimpose → metrics
3) Computes evaluator metrics from `backend/Evaluation/evaluator.py` when deps are available
4) Appends timing + scores to CSV incrementally

Logging: use `--debug` and optional `--log-file`; grep logs for `[EVAL-DEBUG]` or `[EVAL]`.
"""

import sys
import os

# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import csv
import json
import os
import sys
import time
import traceback
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast


# NOTE: `backend/Evaluation/evaluator.py` imports optional heavy deps (e.g. `librosa`).
# We import it lazily inside `main()` so this runner can still export audios + mapping
# even when evaluator deps are missing.
from Variable.configurations import READING_SPEED_WPS, model_config, AUDIO_LDM2, TANGO2
from Variable.dataclases import AudioCue, Cue  
from Tools.decide_audio import decide_audio_cues  
from helper.parallel_audio_generation import parallel_audio_generation  
from helper.audio_conversions import audio_to_base64, dict_to_cue  
from helper.lib import init_models, get_model  
from superimposition_model.superimposition_model import SuperimpositionModel  
import logging

logger = logging.getLogger(__name__)

# Step labels for grep-friendly logs: grep EVAL-DEBUG evaluation_logs.log
_EVAL_STEP = 0


def _eval_debug_reset() -> None:
    global _EVAL_STEP
    _EVAL_STEP = 0


def _eval_debug_step(message: str, **extra: Any) -> None:
    """Structured debug line for each pipeline milestone."""
    global _EVAL_STEP
    _EVAL_STEP += 1
    suffix = ""
    if extra:
        try:
            suffix = " | " + json.dumps(extra, ensure_ascii=False, default=str)
        except Exception:
            suffix = " | " + str(extra)
    logger.info("[EVAL-DEBUG] step=%d %s%s", _EVAL_STEP, message, suffix)


def _log_hf_hub_environment() -> None:
    """
    Log Hugging Face / cache env vars and common footguns (local dir shadowing repo id).
    Helps debug: OSError Can't load config for 'org/repo' ... scheduler_config.json
    """
    keys = (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "XDG_CACHE_HOME",
        "HOME",
    )
    env_snapshot: Dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        if k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN") and v:
            env_snapshot[k] = "<set>"
        else:
            env_snapshot[k] = v
    _eval_debug_step("huggingface_hub_environment", **env_snapshot)

    # Typical Tango2 scheduler repo from training defaults (may appear in errors)
    for repo_id in (
        "sd2-community/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-1",
    ):
        for base in (os.getcwd(), project_root, os.path.join(project_root, "tango_new")):
            candidate = os.path.join(base, repo_id)
            if os.path.isdir(candidate):
                _eval_debug_step(
                    "warning_local_dir_shadows_hf_repo",
                    repo_id=repo_id,
                    path=candidate,
                    hint="Rename/remove this folder or cd elsewhere; diffusers may load it instead of the Hub.",
                )


def _configure_eval_logging(*, debug: bool, log_file: str) -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file.strip():
        log_abs = os.path.abspath(log_file)
        log_dir = os.path.dirname(log_abs)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_abs, encoding="utf-8"))
    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True)
    logging.getLogger(__name__).setLevel(level)

def _cue_dedupe_key(cue: Any) -> tuple:
    """
    Stable-ish key used only for de-duping LLM-returned missing cues.
    """
    a_type = str(getattr(cue, "audio_type", "") or "").upper()
    start_ms = int(getattr(cue, "start_time_ms", 0) or 0)
    duration_ms = int(getattr(cue, "duration_ms", 0) or 0)
    audio_class = getattr(cue, "audio_class", None)
    narrator_description = getattr(cue, "narrator_description", None)
    return (a_type, audio_class, narrator_description, start_ms, duration_ms)


def _missing_items_to_generate_cues(
    missing_items: Any,
    existing_audio_cues: List[Any],
    *,
    skip_audio_types: set[str],
) -> List[Cue]:
    """
    Convert LLM-returned missing-cue items (often list[dict]) into `Cue`s.

    `check_missing_audio_cues()` returns dict objects (your logs show this). The old
    code treated them as strings, so nothing matched and no audio got generated.
    """
    if not missing_items:
        return []

    # Existing cue keys (only used for de-duping; we don't rely on ids)
    existing_keys: set[tuple] = set()
    next_id = 0
    for cw in existing_audio_cues:
        cue = getattr(cw, "audio_cue", None)
        if cue is None:
            continue
        next_id = max(next_id, int(getattr(cue, "id", 0) or 0))
        existing_keys.add(_cue_dedupe_key(cue))

    next_id += 1

    if isinstance(missing_items, dict):
        items_iter = [missing_items]
    else:
        items_iter = missing_items

    out: List[Cue] = []
    for item in items_iter:
        if not isinstance(item, dict):
            # Backwards compatibility: if the model returns audio_class strings,
            # we currently don't have enough info to generate safely.
            continue

        a_type = str(item.get("audio_type", "") or "").upper()
        if a_type in skip_audio_types:
            continue

        cue = dict_to_cue(item)
        if str(getattr(cue, "audio_type", "") or "").upper() in skip_audio_types:
            continue

        key = _cue_dedupe_key(cue)
        if key in existing_keys:
            continue

        # Avoid id collisions for cue->audio generation pipelines.
        cue.id = next_id
        next_id += 1
        out.append(cue)
        existing_keys.add(key)

    return out

@dataclass(frozen=True)
class ExperimentConfig:
    tag: str
    decide_audio_model_name: str
    flags: Dict[str, bool]

    def used_flags_json(self) -> str:
        payload = {
            "tag": self.tag,
            "decide_audio_model_name": self.decide_audio_model_name,
            **self.flags,
        }
        return json.dumps(payload, ensure_ascii=False)


def iter_jsonl(path: str, max_items: Optional[int] = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_items is not None and (idx >= max_items):
                return
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def safe_to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def set_model_config_for_experiment(exp: ExperimentConfig, baseline_flags: Dict[str, bool]) -> None:
    """
    Apply experiment's flags and decide model to the singleton `model_config`.

    We also make sure non-toggled flags remain at baseline.
    """
    model_config.decide_audio_model_name = exp.decide_audio_model_name
    for flag_name, base_val in baseline_flags.items():
        setattr(model_config, flag_name, exp.flags.get(flag_name, base_val))


def _write_header_if_needed(csv_path: str, fieldnames: List[str]) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_row(csv_path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    _write_header_if_needed(csv_path, fieldnames)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, "") for k in fieldnames})

def _hash_short(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _cue_to_dict(cue: Any) -> Dict[str, Any]:
    """
    Convert cue-like objects to plain dict for CSV JSON serialization.
    """
    if hasattr(cue, "__dataclass_fields__"):
        return asdict(cue)
    if hasattr(cue, "model_dump"):
        return cue.model_dump()  # type: ignore[no-any-return]
    if hasattr(cue, "keys"):
        return dict(cue)
    return {
        "id": getattr(cue, "id", None),
        "audio_type": getattr(cue, "audio_type", None),
        "audio_class": getattr(cue, "audio_class", None),
        "start_time_ms": getattr(cue, "start_time_ms", None),
        "duration_ms": getattr(cue, "duration_ms", None),
        "weight_db": getattr(cue, "weight_db", None),
        "fade_ms": getattr(cue, "fade_ms", None),
        "story": getattr(cue, "story", None),
        "narrator_description": getattr(cue, "narrator_description", None),
    }


def _to_json_string(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class SpecialistVariant:
    tag: str
    sfx_model_name: str
    env_model_name: str
    music_model_name: str
    narrator_model_name: str


def build_experiments(
    baseline_flags: Dict[str, bool],
    baseline_decide_audio_model_name: str,
    decide_audio_model_name_variants: List[str],
) -> List[ExperimentConfig]:
    # Baseline
    experiments: List[ExperimentConfig] = [
        ExperimentConfig(
            tag="baseline",
            decide_audio_model_name=baseline_decide_audio_model_name,
            flags=dict(baseline_flags),
        )
    ]


    for model_name in decide_audio_model_name_variants:
        if model_name == baseline_decide_audio_model_name:
            continue
        experiments.append(
            ExperimentConfig(
                tag=f"decide_model_{model_name}",
                decide_audio_model_name=model_name,
                flags=dict(baseline_flags),
            )
        )

    for flag_name, base_val in baseline_flags.items():
        experiments.append(
            ExperimentConfig(
                tag=f"toggle_{flag_name}",
                decide_audio_model_name=baseline_decide_audio_model_name,
                flags={flag_name: (not base_val)},
            )
        )

    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for audio generation pipeline.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "yt_videos",
            "yt_random_test_dataset.jsonl",
        ),
        help="Path to yt_random_test_dataset.jsonl",
    )
    parser.add_argument("--max-items", type=int, default=10, help="Max dataset rows to evaluate.")
    parser.add_argument("--speed-wps", type=float, default=READING_SPEED_WPS, help="Words/sec for cue timing.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional explicit output CSV path. If empty, uses Results/infernce_results/*.csv",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging (DEBUG) and full tracebacks on errors.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Append logs to this file (UTF-8). Example: backend/Evaluation/evaluation_debug.log",
    )
    args = parser.parse_args()

    _configure_eval_logging(debug=bool(args.debug), log_file=str(args.log_file or ""))
    logger.info(
        "[EVAL] start dataset_path=%s max_items=%s speed_wps=%s debug=%s",
        args.dataset_path,
        args.max_items,
        args.speed_wps,
        args.debug,
    )
    _log_hf_hub_environment()

    dataset_path = os.path.abspath(args.dataset_path)
    dataset_filename = os.path.splitext(os.path.basename(dataset_path))[0]


    workspace_root = os.path.dirname(__file__)
    results_dir = os.path.join(workspace_root, "Results", "infernce_results")
    output_csv = (
        os.path.abspath(args.output_csv)
        if args.output_csv
        else os.path.join(results_dir, f"{dataset_filename}_inference_results.csv")
    )
    audio_root_dir = os.path.join(results_dir, "generated_audios")

    bool_flag_names = [
        "fill_coverage_by_llm",
        "use_dsp",
        "use_movie_bgms",
        "use_narrator",
        "use_llm_to_predict_align",
        "use_dsp_to_predict_align",
        "use_avg_llm_and_dsp_to_predict_align",
        "use_dl_based_llm_and_dsp_alignment_predictor",
    ]
    baseline_flags = {name: bool(getattr(model_config, name)) for name in bool_flag_names}
    baseline_decide_audio_model_name = str(model_config.decide_audio_model_name)

    decide_audio_model_name_variants = ["gemini-3-flash-preview", "gemini-2.5-flash"]
    experiments = build_experiments(
        baseline_flags=baseline_flags,
        baseline_decide_audio_model_name=baseline_decide_audio_model_name,
        decide_audio_model_name_variants=decide_audio_model_name_variants,
    )

    # Fieldnames for append-only CSV (stable schema)
    config_fieldnames = [f"flag_{n}" for n in bool_flag_names] + ["decide_audio_model_name", "experiment_tag", "run_type"]
    timing_fieldnames = [
        "cue_decider_seconds",
        "initial_audio_generation_seconds",
        "missing_fill_seconds",
        "superimpose_seconds",
        "audio_to_base64_seconds",
        "pipeline_total_seconds",
        "evaluation_seconds",
        "total_seconds",
    ]
    metric_fieldnames = [
        "clap_score",
        "audio_richness_spectral_flatness",
        "audio_richness_spectral_entropy",
        "noise_floor_db",
        "audio_onsets",
        "yt_coverage_score",
        "yt_sync_score",
        "yt_coverage_and_sync_coverage_score",
        "yt_coverage_and_sync_sync_score",
        "cinematic_dynamic_range_db",
        "cinematic_crest_factor",
        "cinematic_spectral_flatness",
        "cinematic_spectral_entropy",
        "cinematic_spectral_centroid_hz",
        "spectral_kl_divergence",  # skipped by default
        "fad_score",  # skipped by default
        "error",
    ]
    dataset_fieldnames = ["row_index", "source_url", "video_title", "clip_index", "story_prompt"]
    fieldnames = (
        dataset_fieldnames
        + config_fieldnames
        + timing_fieldnames
        + metric_fieldnames
        + [
            "used_flags_json",
            "sfx_model_name",
            "env_model_name",
            "music_model_name",
            "narrator_model_name",
            "specialist_model_variant_tag",
            "llm_suggested_cues_json",
            "final_superimposed_cues_json",
            "audio_wav_path",
            "audio_export_error",
        ]
    )

    # Preload heavy models once (Tango2 / Parler / AudioLDM2 may hit Hugging Face here).
    _eval_debug_step("before_init_models")
    logger.info("[EVAL] Preloading specialist models (init_models)...")
    try:
        init_models()
    except OSError as e:
        logger.exception(
            "[EVAL] init_models OSError (often Hugging Face Hub: missing scheduler_config.json, "
            "offline mode, gated repo, or a local directory shadowing the model id). Original error: %s",
            e,
        )
        raise
    except Exception:
        logger.exception(
            "[EVAL] init_models failed (see traceback; AudioLDM2/Tango2/diffusers load here)."
        )
        raise
    _eval_debug_step("after_init_models_ok")

    evaluator: Any = None
    try:
        _eval_debug_step("before_audio_evaluator_init")
        logger.info("[EVAL] Initializing evaluator (loads CLAP + embeddings)...")
        from Evaluation.evaluator import AudioEvaluator  # local/lazy import

        evaluator = AudioEvaluator()
        _eval_debug_step("after_audio_evaluator_init_ok")
    except Exception as e:
        # Allow audio export + mapping even if evaluator deps are missing.
        evaluator = None
        logger.warning("[EVAL] Evaluator disabled due to import/init error: %s", e, exc_info=args.debug)
    superimposition_model_ins = SuperimpositionModel()
    _eval_debug_step("after_superimposition_model_init")

    # Build specialist model variants (baseline + one-at-a-time AudioLDM2 toggles).
    baseline_specialist = SpecialistVariant(
        tag="baseline_tango2",
        sfx_model_name=TANGO2,
        env_model_name=TANGO2,
        music_model_name=TANGO2,
        narrator_model_name=model_config.narrator_model_name,
    )
    specialist_variants: List[SpecialistVariant] = [
        baseline_specialist,
        SpecialistVariant(
            tag="toggle_sfx_audioldm2",
            sfx_model_name=AUDIO_LDM2,
            env_model_name=TANGO2,
            music_model_name=TANGO2,
            narrator_model_name=model_config.narrator_model_name,
        ),
        SpecialistVariant(
            tag="toggle_env_audioldm2",
            sfx_model_name=TANGO2,
            env_model_name=AUDIO_LDM2,
            music_model_name=TANGO2,
            narrator_model_name=model_config.narrator_model_name,
        ),
        SpecialistVariant(
            tag="toggle_music_audioldm2",
            sfx_model_name=TANGO2,
            env_model_name=TANGO2,
            music_model_name=AUDIO_LDM2,
            narrator_model_name=model_config.narrator_model_name,
        ),
    ]

    # Evaluate dataset incrementally.
    for row_index, record in iter_jsonl(dataset_path, max_items=args.max_items):
        _eval_debug_reset()
        _eval_debug_step(
            "dataset_row_start",
            row_index=row_index,
            source_url=str(record.get("source_url", ""))[:120],
        )
        story_prompt = str(record.get("story_prompt") or "").strip()
        if not story_prompt:
            # Still write a row for each experiment so the CSV schema stays stable.
            for exp in experiments:
                # Use baseline specialist config for skipped rows.
                row = {
                    "row_index": row_index,
                    "source_url": record.get("source_url", ""),
                    "video_title": record.get("video_title", ""),
                    "clip_index": record.get("clip_index", ""),
                    "story_prompt": "",
                    "decide_audio_model_name": exp.decide_audio_model_name,
                    "experiment_tag": exp.tag,
                    "run_type": "skip_empty_story_prompt",
                    **{f"flag_{n}": exp.flags.get(n, baseline_flags[n]) for n in bool_flag_names},
                    "used_flags_json": exp.used_flags_json(),
                    **{k: 0.0 for k in timing_fieldnames},
                    **{k: "" for k in metric_fieldnames},
                    "sfx_model_name": baseline_specialist.sfx_model_name,
                    "env_model_name": baseline_specialist.env_model_name,
                    "music_model_name": baseline_specialist.music_model_name,
                    "narrator_model_name": baseline_specialist.narrator_model_name,
                    "specialist_model_variant_tag": baseline_specialist.tag,
                    "llm_suggested_cues_json": "[]",
                    "final_superimposed_cues_json": "[]",
                    "audio_wav_path": "",
                    "audio_export_error": "",
                    "error": "Empty story_prompt; skipping pipeline.",
                }
                append_row(output_csv, fieldnames, row)
            logger.info("[EVAL] row=%d skipped: empty story_prompt", row_index)
            continue

        for exp in experiments:
            _eval_debug_step(
                "experiment_start",
                row_index=row_index,
                experiment_tag=exp.tag,
                decide_audio_model_name=exp.decide_audio_model_name,
            )
            # Apply config to singleton
            used_flags_resolved = {**baseline_flags, **exp.flags}
            run_type = exp.tag
            set_model_config_for_experiment(exp, baseline_flags=baseline_flags)

            pipeline_start = time.perf_counter()
            stage_decide = 0.0
            # We will accumulate per-variant metrics inside the inner loop; here we only
            # measure the cue-decider once.
            llm_suggested_cues_json = "[]"

            try:
                # Step 1: decide audio cues (LLM)
                _eval_debug_step(
                    "before_decide_audio_cues",
                    use_narrator=model_config.use_narrator,
                    use_movie_bgms=model_config.use_movie_bgms,
                    decide_audio_model_name=model_config.decide_audio_model_name,
                )
                t0 = time.perf_counter()
                cues, total_duration_ms = decide_audio_cues(
                    story_prompt,
                    args.speed_wps,
                    narrator_enabled=model_config.use_narrator,
                    movie_bgms_enabled=model_config.use_movie_bgms,
                )
                llm_suggested_cues_json = _to_json_string([_cue_to_dict(c) for c in cues])
                stage_decide = time.perf_counter() - t0
                _eval_debug_step(
                    "after_decide_audio_cues",
                    cue_count=len(cues),
                    total_duration_ms=total_duration_ms,
                    cue_decider_seconds=round(stage_decide, 4),
                )
                # Step 2+: for each specialist variant, generate audio using SAME cues.
                for spec_variant in specialist_variants:
                    _eval_debug_step(
                        "specialist_variant_start",
                        tag=spec_variant.tag,
                        sfx=spec_variant.sfx_model_name,
                        env=spec_variant.env_model_name,
                        music=spec_variant.music_model_name,
                        narrator=spec_variant.narrator_model_name,
                    )
                    # Reset specialist model names for this variant.
                    model_config.sfx_model_name = spec_variant.sfx_model_name
                    model_config.env_model_name = spec_variant.env_model_name
                    model_config.music_model_name = spec_variant.music_model_name
                    model_config.narrator_model_name = spec_variant.narrator_model_name

                    stage_initial_gen = 0.0
                    stage_missing_fill = 0.0
                    stage_superimpose = 0.0
                    stage_audio_to_base64 = 0.0
                    stage_eval = 0.0
                    stage_total = 0.0
                    metrics: Dict[str, Any] = {}
                    audio_wav_path = ""
                    audio_export_error = ""
                    final_superimposed_cues_json = "[]"
                    variant_error = ""

                    t_variant_start = time.perf_counter()

                    # Validate selected specialist models before generation.
                    # If a model is not registered/available, record a clean error row for this variant.
                    _eval_debug_step("before_get_model_validation")
                    try:
                        get_model(spec_variant.sfx_model_name)
                        get_model(spec_variant.env_model_name)
                        get_model(spec_variant.music_model_name)
                        _eval_debug_step("after_get_model_validation_ok")
                    except Exception as e:
                        variant_error = f"Specialist model unavailable: {e}"
                        row = {
                            "row_index": row_index,
                            "source_url": record.get("source_url", ""),
                            "video_title": record.get("video_title", ""),
                            "clip_index": record.get("clip_index", ""),
                            "story_prompt": story_prompt[:500],
                            "decide_audio_model_name": exp.decide_audio_model_name,
                            "experiment_tag": exp.tag,
                            "run_type": run_type,
                            **{f"flag_{n}": used_flags_resolved[n] for n in bool_flag_names},
                            "used_flags_json": exp.used_flags_json(),
                            "cue_decider_seconds": stage_decide,
                            "initial_audio_generation_seconds": 0.0,
                            "missing_fill_seconds": 0.0,
                            "superimpose_seconds": 0.0,
                            "audio_to_base64_seconds": 0.0,
                            "pipeline_total_seconds": time.perf_counter() - pipeline_start,
                            "evaluation_seconds": 0.0,
                            "total_seconds": time.perf_counter() - t_variant_start,
                            **{k: "" for k in metric_fieldnames},
                            "sfx_model_name": spec_variant.sfx_model_name,
                            "env_model_name": spec_variant.env_model_name,
                            "music_model_name": spec_variant.music_model_name,
                            "narrator_model_name": spec_variant.narrator_model_name,
                            "specialist_model_variant_tag": spec_variant.tag,
                            "llm_suggested_cues_json": llm_suggested_cues_json,
                            "final_superimposed_cues_json": "[]",
                            "audio_wav_path": "",
                            "audio_export_error": "",
                            "error": variant_error,
                        }
                        append_row(output_csv, fieldnames, row)
                        logger.warning(
                            "Skipping variant %s due to unavailable model(s): %s",
                            spec_variant.tag,
                            e,
                            exc_info=args.debug,
                        )
                        continue

                    # Step 2: generate cue audio (specialists)
                    _eval_debug_step(
                        "before_parallel_audio_generation",
                        cue_count=len(cues),
                        note="Tango2/AudioLDM2 may call Hugging Face (scheduler/unet) on first real batch.",
                    )
                    logger.info(
                        "[EVAL] row=%d exp=%s variant=%s parallel_audio_generation n_cues=%d",
                        row_index,
                        exp.tag,
                        spec_variant.tag,
                        len(cues),
                    )
                    t1 = time.perf_counter()
                    try:
                        audio_cues = parallel_audio_generation(cast(List[Cue], list(cues)))
                    except OSError as gen_e:
                        logger.exception(
                            "[EVAL] parallel_audio_generation OSError row=%d variant=%s (HF config/cache/repo?): %s",
                            row_index,
                            spec_variant.tag,
                            gen_e,
                        )
                        raise
                    stage_initial_gen = time.perf_counter() - t1
                    _eval_debug_step(
                        "after_parallel_audio_generation",
                        wrapped_cue_count=len(audio_cues),
                        seconds=round(stage_initial_gen, 4),
                    )

                    # Step 3 (optional): fill missing coverage
                    if model_config.fill_coverage_by_llm and audio_cues:
                        _eval_debug_step(
                            "before_missing_coverage_fill",
                            fill_coverage_by_llm=True,
                            existing_wrapped_cues=len(audio_cues),
                        )
                        t2 = time.perf_counter()
                        not_covered_classes = superimposition_model_ins.check_missing_audio_cues(
                            story_prompt, audio_cues, total_duration_ms
                        )
                        if not_covered_classes:
                            # `check_missing_audio_cues()` returns LLM-generated cue dicts (not strings),
                            # so we convert them to `Cue` objects and then generate audio.
                            missing_cues_to_generate = _missing_items_to_generate_cues(
                                not_covered_classes,
                                audio_cues,
                                skip_audio_types={"NARRATOR"},
                            )
                            generated_missing = parallel_audio_generation(
                                cast(List[Cue], missing_cues_to_generate)
                            )
                            logger.info(
                                "Missing cue fill: candidates=%d to_generate=%d generated=%d",
                                len(not_covered_classes) if isinstance(not_covered_classes, list) else 1,
                                len(missing_cues_to_generate),
                                len(generated_missing),
                            )
                            audio_cues.extend(generated_missing)
                            

                        stage_missing_fill = time.perf_counter() - t2
                        _eval_debug_step(
                            "after_missing_coverage_fill",
                            seconds=round(stage_missing_fill, 4),
                            final_wrapped_cue_count=len(audio_cues),
                        )
                    else:
                        _eval_debug_step(
                            "skip_missing_coverage_fill",
                            fill_coverage_by_llm=model_config.fill_coverage_by_llm,
                            has_audio_cues=bool(audio_cues),
                        )

                    # Step 4: superimpose to final audio
                    _eval_debug_step(
                        "before_superimpose",
                        use_dsp=model_config.use_dsp,
                        wrapped_cue_count=len(audio_cues),
                        total_duration_ms=total_duration_ms,
                    )
                    t3 = time.perf_counter()
                    final_audio = superimposition_model_ins.superimpose_audio_cues_with_audio_base64(
                        story_prompt, audio_cues, total_duration_ms
                    )
                    stage_superimpose = time.perf_counter() - t3
                    _eval_debug_step(
                        "after_superimpose",
                        seconds=round(stage_superimpose, 4),
                        final_audio_len_ms=len(final_audio),
                    )

                    # Export generated audio for later inspection.
                    flags_hash = _hash_short(exp.used_flags_json() + "|" + story_prompt + "|" + spec_variant.tag)
                    audio_subdir = os.path.join(
                        audio_root_dir,
                        exp.tag,
                        exp.decide_audio_model_name,
                        spec_variant.tag,
                    )
                    os.makedirs(audio_subdir, exist_ok=True)
                    audio_filename = f"row_{row_index}_{flags_hash}.wav"
                    audio_wav_path = os.path.join(audio_subdir, audio_filename)
                    _eval_debug_step("before_wav_export", path=audio_wav_path)
                    try:
                        final_audio.export(audio_wav_path, format="wav")
                    except Exception as export_e:
                        audio_export_error = str(export_e)
                        logger.warning(
                            "[EVAL] wav export failed row=%d variant=%s: %s",
                            row_index,
                            spec_variant.tag,
                            export_e,
                            exc_info=args.debug,
                        )
                    _eval_debug_step(
                        "after_wav_export",
                        audio_wav_path=audio_wav_path,
                        audio_export_error=audio_export_error or None,
                    )

                    # Step 5: convert to base64
                    _eval_debug_step("before_audio_to_base64")
                    t4 = time.perf_counter()
                    audio_base64 = audio_to_base64(final_audio)
                    stage_audio_to_base64 = time.perf_counter() - t4
                    _eval_debug_step(
                        "after_audio_to_base64",
                        seconds=round(stage_audio_to_base64, 4),
                        b64_len=len(audio_base64) if audio_base64 else 0,
                    )

                    pipeline_total = time.perf_counter() - pipeline_start

                    audio_cues_final: List[Any] = audio_cues
                    final_superimposed_cues_json = _to_json_string(
                        [_cue_to_dict(c.audio_cue) for c in audio_cues_final]
                    )

                    # Step 6: evaluate
                    _eval_debug_step(
                        "before_metrics",
                        evaluator_active=evaluator is not None,
                        yt_audio_cue_count=len(
                            [c for c in audio_cues_final if isinstance(c.audio_cue, AudioCue)]
                        ),
                    )
                    t5 = time.perf_counter()
                    yt_audio_cues: List[AudioCue] = [
                        c.audio_cue for c in audio_cues_final if isinstance(c.audio_cue, AudioCue)
                    ]

                    if evaluator is not None:
                        # CLAP + general audio metrics
                        metrics["clap_score"] = evaluator.get_clap_score(audio_base64, story_prompt)
                        flatness, spec_entropy = evaluator.get_audio_richness(audio_base64)
                        metrics["audio_richness_spectral_flatness"] = flatness
                        metrics["audio_richness_spectral_entropy"] = spec_entropy
                        metrics["noise_floor_db"] = evaluator.get_noise_floor(audio_base64)
                        metrics["audio_onsets"] = evaluator.evaluate_sync_from_audio_base64(audio_base64)

                        # YT coverage/sync metrics
                        metrics["yt_coverage_score"] = evaluator.yt_coverage_score(story_prompt, yt_audio_cues)
                        metrics["yt_sync_score"] = evaluator.yt_sync_score(story_prompt, yt_audio_cues)
                        yt_cov_sync = evaluator.yt_coverage_and_sync_score(story_prompt, yt_audio_cues)
                        metrics["yt_coverage_and_sync_coverage_score"] = yt_cov_sync.get("coverage_score", "")
                        metrics["yt_coverage_and_sync_sync_score"] = yt_cov_sync.get("sync_score", "")

                        cinematic = evaluator.get_cinematic_acoustic_metrics(audio_base64) or {}
                        metrics["cinematic_dynamic_range_db"] = cinematic.get("dynamic_range_db", "")
                        metrics["cinematic_crest_factor"] = cinematic.get("crest_factor", "")
                        metrics["cinematic_spectral_flatness"] = cinematic.get("spectral_flatness", "")
                        metrics["cinematic_spectral_entropy"] = cinematic.get("spectral_entropy", "")
                        metrics["cinematic_spectral_centroid_hz"] = cinematic.get("spectral_centroid_hz", "")

                        # KL divergence and FAD are skipped (no reference generated locally).
                        metrics["spectral_kl_divergence"] = ""
                        metrics["fad_score"] = ""

                        stage_eval = time.perf_counter() - t5
                        metrics["error"] = ""
                    else:
                        stage_eval = time.perf_counter() - t5
                        for k in metric_fieldnames:
                            if k in metrics:
                                continue
                            if k in {"spectral_kl_divergence", "fad_score"}:
                                metrics[k] = ""
                            else:
                                metrics[k] = ""
                        metrics["error"] = "Evaluator disabled (missing deps). Audio exported, metrics skipped."

                    stage_total = time.perf_counter() - t_variant_start
                    _eval_debug_step(
                        "after_metrics",
                        evaluation_seconds=round(stage_eval, 4),
                        clap_preview=str(metrics.get("clap_score", ""))[:80],
                    )

                    # Append row for this specialist variant
                    row = {
                        "row_index": row_index,
                        "source_url": record.get("source_url", ""),
                        "video_title": record.get("video_title", ""),
                        "clip_index": record.get("clip_index", ""),
                        "story_prompt": story_prompt[:500],
                        "decide_audio_model_name": exp.decide_audio_model_name,
                        "experiment_tag": exp.tag,
                        "run_type": run_type,
                        **{f"flag_{n}": used_flags_resolved[n] for n in bool_flag_names},
                        "used_flags_json": exp.used_flags_json(),
                        "cue_decider_seconds": stage_decide,
                        "initial_audio_generation_seconds": stage_initial_gen,
                        "missing_fill_seconds": stage_missing_fill,
                        "superimpose_seconds": stage_superimpose,
                        "audio_to_base64_seconds": stage_audio_to_base64,
                        "pipeline_total_seconds": pipeline_total,
                        "evaluation_seconds": stage_eval,
                        "total_seconds": stage_total,
                        **metrics,
                        "sfx_model_name": spec_variant.sfx_model_name,
                        "env_model_name": spec_variant.env_model_name,
                        "music_model_name": spec_variant.music_model_name,
                        "narrator_model_name": spec_variant.narrator_model_name,
                        "specialist_model_variant_tag": spec_variant.tag,
                        "llm_suggested_cues_json": llm_suggested_cues_json,
                        "final_superimposed_cues_json": final_superimposed_cues_json,
                        "audio_wav_path": audio_wav_path,
                        "audio_export_error": audio_export_error,
                    }

                    append_row(output_csv, fieldnames, row)
                    logger.info(
                        "[EVAL] row=%d exp=%s variant=%s OK pipeline_total=%.2fs wav=%s",
                        row_index,
                        exp.tag,
                        spec_variant.tag,
                        pipeline_total,
                        audio_wav_path or "(none)",
                    )
                    _eval_debug_step("specialist_variant_complete", tag=spec_variant.tag)
            except Exception as e:
                # Append an error row to keep datasets comparable.
                logger.exception(
                    "[EVAL] pipeline error row=%d exp=%s type=%s: %s",
                    row_index,
                    exp.tag,
                    type(e).__name__,
                    e,
                )
                pipeline_total = time.perf_counter() - pipeline_start
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                # Even on error, record one row with baseline specialist info.
                row = {
                    "row_index": row_index,
                    "source_url": record.get("source_url", ""),
                    "video_title": record.get("video_title", ""),
                    "clip_index": record.get("clip_index", ""),
                    "story_prompt": story_prompt[:500],
                    "decide_audio_model_name": exp.decide_audio_model_name,
                    "experiment_tag": exp.tag,
                    "run_type": run_type,
                    **{f"flag_{n}": used_flags_resolved[n] for n in bool_flag_names},
                    "used_flags_json": exp.used_flags_json(),
                    "cue_decider_seconds": stage_decide,
                    "initial_audio_generation_seconds": 0.0,
                    "missing_fill_seconds": 0.0,
                    "superimpose_seconds": 0.0,
                    "audio_to_base64_seconds": 0.0,
                    "pipeline_total_seconds": pipeline_total,
                    "evaluation_seconds": 0.0,
                    "total_seconds": pipeline_total,
                    **{k: "" for k in metric_fieldnames},
                    "sfx_model_name": baseline_specialist.sfx_model_name,
                    "env_model_name": baseline_specialist.env_model_name,
                    "music_model_name": baseline_specialist.music_model_name,
                    "narrator_model_name": baseline_specialist.narrator_model_name,
                    "specialist_model_variant_tag": baseline_specialist.tag,
                    "llm_suggested_cues_json": llm_suggested_cues_json,
                    "final_superimposed_cues_json": "[]",
                    "audio_wav_path": "",
                    "audio_export_error": "",
                    "error": err or str(e),
                }
                append_row(output_csv, fieldnames, row)

            logger.info(
                f"[row {row_index}] exp={exp.tag} decide_model={exp.decide_audio_model_name} "
                f"cue_decider={stage_decide:.2f}s"
            )


if __name__ == "__main__":
    main()

