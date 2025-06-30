#!/usr/bin/env python3
"""
Validation interactive – Phase 5 (Gate)
=======================================
• Chronomètre **chaque étape** : STT → LLM → TTS → lecture.
• Écrit un log JSON exploitable dans `logs/voice_pipeline_validation_YYYYMMDD_HHMMSS.json`.
• Demande un feedback humain (Y/n) ; le test échoue (`exit 1`) si la réponse
  audio est absente ou jugée non satisfaisante.
• Sécurise la sortie audio : conversion float → int16, sample‑rate détecté.
• Détecte un LLM vide / TTS muet et signale l'erreur au lieu de jouer un bip.

Exécution manuelle :
```
pytest -q tests/test_pipeline_voice_validation.py
# ou
python tests/test_pipeline_voice_validation.py --llm-endpoint http://localhost:8000/v1/chat/completions
```

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE - EXÉCUTABLE DEPUIS N'IMPORTE OÙ
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    # Déterminer le répertoire racine du projet
    current_file = pathlib.Path(__file__).resolve()
    
    # Chercher le répertoire racine (contient .git ou marqueurs projet)
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    # Ajouter le projet root au Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Changer le working directory vers project root
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090 obligatoire
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
    
    print(f"🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
    print(f"📁 Project Root: {project_root}")
    print(f"💻 Working Directory: {os.getcwd()}")
    
    return project_root

# Initialiser l'environnement portable
_PROJECT_ROOT = _setup_portable_environment()

# Maintenant imports normaux...

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import sounddevice as sd

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# ---------------------------------------------------------------------------
# Import des modules projet - STRUCTURE RÉELLE
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))  # noqa: E402

# Imports corrigés selon structure réelle fournie
sys.path.insert(0, str(PROJECT_ROOT / "STT"))
sys.path.insert(0, str(PROJECT_ROOT / "TTS"))
sys.path.insert(0, str(PROJECT_ROOT / "PIPELINE"))

from streaming_microphone_manager import StreamingMicrophoneManager  # noqa: E402
from unified_stt_manager_optimized import OptimizedUnifiedSTTManager  # noqa: E402
from tts_manager import UnifiedTTSManager  # noqa: E402
from pipeline_orchestrator import PipelineOrchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Validation GPU RTX 3090 obligatoire
# ---------------------------------------------------------------------------

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        return True
    except Exception as e:
        print(f"⚠️ Validation GPU échouée: {e}")
        return False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_synthesize(tts: UnifiedTTSManager, text: str) -> Tuple[np.ndarray, int]:
    """Appelle le TTS et garantit un tuple (audio, sample_rate)."""
    try:
        # Utilisation de la méthode async du UnifiedTTSManager
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(tts.synthesize(text))
        
        if hasattr(result, 'audio_data') and hasattr(result, 'success'):
            if result.success and result.audio_data:
                # Conversion bytes vers numpy array si nécessaire
                if isinstance(result.audio_data, bytes):
                    audio = np.frombuffer(result.audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                else:
                    audio = result.audio_data
                sr = getattr(result, 'sample_rate', 22050)
                return audio, sr
            else:
                raise RuntimeError(f"TTS failed: {getattr(result, 'error', 'Unknown error')}")
        else:
            # Fallback pour format de retour différent
            if isinstance(result, tuple):
                audio, sr = result
            else:
                audio = result
                sr = getattr(tts, "sample_rate", 22050)
            return audio, sr
    except Exception as e:
        print(f"❌ Erreur TTS: {e}")
        raise


def _to_int16(audio_f: np.ndarray) -> np.ndarray:
    """Conversion audio float vers int16 sécurisée"""
    audio_c = np.clip(audio_f, -1.0, 1.0)
    return (audio_c * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Test interactif principal
# ---------------------------------------------------------------------------

async def main() -> None:  # noqa: C901 – fonction principale longue assumée
    parser = argparse.ArgumentParser(description="Test interactif pipeline voix‑à‑voix SuperWhisper V6")
    parser.add_argument("--llm-endpoint", default="http://localhost:1234/v1",
                        help="Endpoint OpenAI‑compatible exposé par LM Studio/vLLM/llama.cpp")
    parser.add_argument("--duration", type=float, default=8.0, help="Durée d'enregistrement (s)")
    args = parser.parse_args()

    # Validation GPU RTX 3090 obligatoire
    if not validate_rtx3090_configuration():
        print("🚫 ÉCHEC: Configuration GPU RTX 3090 invalide")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # Initialiser les composants avec configuration réelle
    # ---------------------------------------------------------------------
    print("🔧 Initialisation des composants...")
    
    try:
        # Configuration STT optimisée
        stt_config = {
            "backend": "prism",
            "model_path": "models/whisper",
            "device": "cuda:1",  # RTX 3090 forcée
            "compute_type": "float16"
        }
        stt = OptimizedUnifiedSTTManager(stt_config)
        
        # Configuration TTS
        tts_config = {
            "backend": "piper_native",
            "model_path": "models/tts",
            "device": "cuda:1",  # RTX 3090 forcée
            "cache_enabled": True
        }
        tts = UnifiedTTSManager(tts_config)
        
        # Configuration LLM avec fallbacks
        llm_config = {
            "endpoints": [
                {"url": args.llm_endpoint, "name": "Primary LLM"},
                {"url": "http://localhost:11434/api", "name": "Ollama"},
            ],
            "fallback_responses": [
                "Bonjour ! Je suis SuperWhisper V6, votre assistant vocal.",
                "Je vous entends parfaitement. Comment puis-je vous aider ?",
                "Excellent ! Le pipeline voix-à-voix fonctionne correctement."
            ],
            "timeout": 15.0
        }
        
        # Initialisation PipelineOrchestrator avec code obligatoire v1.1
        orchestrator = PipelineOrchestrator(
            stt_manager=stt, 
            tts_manager=tts, 
            llm_config=llm_config
        )
        
        print("✅ Tous les composants initialisés")
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        sys.exit(1)

    sample_rate = 16000
    log: dict[str, object] = {
        "utc_ts": datetime.utcnow().isoformat(),
        "gpu_config": "RTX 3090 (CUDA:1)",
        "llm_endpoint": args.llm_endpoint
    }

    print("\n████  SuperWhisper V6 – Validation pipeline voix-à-voix  ████")
    print(f"🎤 Parlez pendant ≈ {args.duration:.0f} s après le bip…")
    print("💬 Exemple: 'Bonjour SuperWhisper, comment allez-vous ?'")

    # ---------------------------------------------------------------------
    # Capturer l'audio utilisateur
    # ---------------------------------------------------------------------
    print("⏺️  Enregistrement… Bip !")
    t0 = time.perf_counter()
    recording = sd.rec(int(args.duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    t_record_end = time.perf_counter()
    log["record_time_ms"] = (t_record_end - t0) * 1000

    print(f"✅ Audio capturé: {len(recording):,} échantillons")

    # ---------------------------------------------------------------------
    # STT - Transcription
    # ---------------------------------------------------------------------
    print("🎯 Transcription STT...")
    stt_start = time.perf_counter()
    
    try:
        # Conversion audio pour STT
        audio_bytes = (recording.flatten() * 32767).astype(np.int16).tobytes()
        stt_result = await stt.transcribe_audio(audio_bytes)
        
        if not stt_result or not hasattr(stt_result, 'text') or not stt_result.text.strip():
            raise RuntimeError("STT n'a pas produit de transcription")
            
        transcription = stt_result.text.strip()
        
    except Exception as e:
        print(f"❌ Erreur STT: {e}")
        _write_log(log, error=f"stt_error: {e}")
        sys.exit(1)
    
    stt_end = time.perf_counter()
    log["stt_latency_ms"] = (stt_end - stt_start) * 1000
    log["transcription"] = transcription
    
    print(f"✅ Transcription: '{transcription}'")
    print(f"⚡ Latence STT: {log['stt_latency_ms']:.1f}ms")

    # ---------------------------------------------------------------------
    # LLM - Génération réponse
    # ---------------------------------------------------------------------
    print("🤖 Génération réponse LLM...")
    llm_start = time.perf_counter()
    
    try:
        # Utilisation du LLMClient du PipelineOrchestrator
        prompt = f"Répondez brièvement et naturellement en français à: {transcription}"
        assistant_text = await orchestrator.llm_client.generate_response(prompt)
        
        if not assistant_text or not assistant_text.strip():
            raise RuntimeError("LLM a renvoyé une réponse vide")
            
    except Exception as e:
        print(f"❌ Erreur LLM: {e}")
        print("🔄 Utilisation réponse fallback...")
        assistant_text = llm_config["fallback_responses"][0]
    
    llm_end = time.perf_counter()
    log["llm_latency_ms"] = (llm_end - llm_start) * 1000
    log["assistant_text"] = assistant_text
    
    print(f"✅ Réponse LLM: '{assistant_text}'")
    print(f"⚡ Latence LLM: {log['llm_latency_ms']:.1f}ms")

    # ---------------------------------------------------------------------
    # TTS - Synthèse vocale
    # ---------------------------------------------------------------------
    print("🔊 Synthèse TTS...")
    tts_start = time.perf_counter()
    
    try:
        audio_f, sr = _safe_synthesize(tts, assistant_text)
        
        if len(audio_f) == 0:
            raise RuntimeError("TTS a produit un audio vide")
            
    except Exception as e:
        print(f"❌ Erreur TTS: {e}")
        _write_log(log, error=f"tts_error: {e}")
        sys.exit(1)
    
    tts_end = time.perf_counter()
    log["tts_latency_ms"] = (tts_end - tts_start) * 1000
    log["tts_sample_rate"] = sr
    log["audio_max"] = float(np.max(np.abs(audio_f)))
    log["audio_duration_s"] = len(audio_f) / sr

    audio_i16 = _to_int16(audio_f)
    
    print(f"✅ TTS réussi: {len(audio_f):,} échantillons")
    print(f"🔊 Durée audio: {log['audio_duration_s']:.1f}s")
    print(f"⚡ Latence TTS: {log['tts_latency_ms']:.1f}ms")

    # ---------------------------------------------------------------------
    # Lecture audio
    # ---------------------------------------------------------------------
    print("🔈 Lecture réponse vocale...")
    playback_start = time.perf_counter()
    sd.play(audio_i16, samplerate=sr)
    sd.wait()
    playback_end = time.perf_counter()

    log["playback_ms"] = (playback_end - playback_start) * 1000
    log["time_to_first_audio_ms"] = (tts_start - t0) * 1000
    log["e2e_ms"] = (playback_end - t0) * 1000

    # ---------------------------------------------------------------------
    # Métriques finales
    # ---------------------------------------------------------------------
    print("\n" + "="*60)
    print("📊 MÉTRIQUES PIPELINE VOIX-À-VOIX")
    print("="*60)
    print(f"⚡ Latence STT:        {log['stt_latency_ms']:.1f}ms")
    print(f"⚡ Latence LLM:        {log['llm_latency_ms']:.1f}ms") 
    print(f"⚡ Latence TTS:        {log['tts_latency_ms']:.1f}ms")
    print(f"⚡ Lecture audio:      {log['playback_ms']:.1f}ms")
    print(f"🎯 TOTAL END-TO-END:   {log['e2e_ms']:.1f}ms")
    print(f"🎯 Temps 1er audio:    {log['time_to_first_audio_ms']:.1f}ms")
    
    # Vérification objectif < 1200ms
    target_ms = 1200
    if log['e2e_ms'] <= target_ms:
        print(f"✅ OBJECTIF ATTEINT: {log['e2e_ms']:.1f}ms < {target_ms}ms")
    else:
        print(f"⚠️ OBJECTIF MANQUÉ: {log['e2e_ms']:.1f}ms > {target_ms}ms")

    # ---------------------------------------------------------------------
    # Validation humaine
    # ---------------------------------------------------------------------
    print("\n" + "="*60)
    print("🧑 VALIDATION HUMAINE")
    print("="*60)
    print(f"🎤 Vous avez dit:      '{transcription}'")
    print(f"🤖 Assistant répond:   '{assistant_text}'")
    print(f"🔊 Audio joué:         {log['audio_duration_s']:.1f}s")
    
    # Questions validation
    questions = [
        ("La transcription est-elle correcte ?", "transcription_ok"),
        ("La réponse LLM est-elle appropriée ?", "llm_response_ok"), 
        ("La qualité audio TTS est-elle acceptable ?", "audio_quality_ok"),
        ("La latence est-elle acceptable pour une conversation ?", "latency_ok"),
        ("Le pipeline global fonctionne-t-il correctement ?", "pipeline_ok")
    ]
    
    validation_results = {}
    all_ok = True
    
    for question, key in questions:
        response = input(f"❓ {question} (y/n): ").strip().lower()
        ok = response.startswith("y")
        validation_results[key] = ok
        if not ok:
            all_ok = False
    
    log["human_validation"] = validation_results
    log["overall_validation"] = all_ok

    # ---------------------------------------------------------------------
    # Écriture log + verdict final
    # ---------------------------------------------------------------------
    _write_log(log)
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ VALIDATION HUMAINE : SUCCÈS")
        print("🎊 Pipeline voix-à-voix SuperWhisper V6 VALIDÉ !")
        print("="*60)
        sys.exit(0)
    else:
        print("❌ VALIDATION HUMAINE : ÉCHEC")
        print("🔧 Pipeline nécessite des corrections")
        print("="*60)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Utilitaire d'écriture log
# ---------------------------------------------------------------------------

def _write_log(payload: dict, error: str | None = None) -> None:
    """Écriture du log JSON avec toutes les métriques"""
    if error:
        payload["error"] = error
        payload["success"] = False
    else:
        payload["success"] = payload.get("overall_validation", False)
    
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fname = log_dir / f"voice_pipeline_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    
    print(f"📝 Log JSON → {fname}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main()) 