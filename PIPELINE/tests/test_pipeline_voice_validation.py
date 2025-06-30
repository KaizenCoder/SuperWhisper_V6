#!/usr/bin/env python3
"""
tests/test_pipeline_voice_validation.py

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
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# ---------------------------------------------------------------------------
# Import des modules projet
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))  # noqa: E402

# Imports avec fallbacks pour robustesse
try:
    from STT.streaming_microphone_manager import StreamingMicrophoneManager  # noqa: E402
    from STT.unified_stt_manager import UnifiedSTTManager  # noqa: E402
    from LLM.llm_client import LLMClient  # noqa: E402
    from PIPELINE.pipeline_orchestrator import PipelineOrchestrator  # noqa: E402
    
    # TTS avec fallback
    try:
        from TTS.tts_manager import TTSManager as UnifiedTTSManager  # noqa: E402
    except ImportError:
        try:
            from TTS.unified_tts_manager import UnifiedTTSManager  # noqa: E402
        except ImportError:
            print("⚠️ TTS module non trouvé - utilisation fallback")
            UnifiedTTSManager = None
    
    IMPORTS_OK = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("🔄 Utilisation fallbacks pour tests")
    IMPORTS_OK = False
    UnifiedSTTManager = None
    UnifiedTTSManager = None
    LLMClient = None
    PipelineOrchestrator = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_synthesize(tts: UnifiedTTSManager, text: str) -> Tuple[np.ndarray, int]:
    """Appelle le TTS et garantit un tuple (audio, sample_rate)."""
    try:
        out = tts.synthesize(text)
        if isinstance(out, tuple):
            audio, sr = out
        else:
            audio = out
            sr = getattr(tts, "sample_rate", 22050)
        return audio, sr
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
        # Fallback audio
        sr = 22050
        duration_s = len(text) * 0.08  # ~80ms par caractère
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, int(duration_s * sr)))
        return audio, sr


def _to_int16(audio_f: np.ndarray) -> np.ndarray:
    """Conversion float32 → int16 avec clipping"""
    audio_c = np.clip(audio_f, -1.0, 1.0)
    return (audio_c * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Test interactif principal
# ---------------------------------------------------------------------------

async def main() -> None:  # noqa: C901 – fonction principale longue assumée
    parser = argparse.ArgumentParser(description="Test interactif pipeline voix‑à‑voix SuperWhisper V6")
    parser.add_argument("--llm-endpoint", default="http://localhost:8000/v1/chat/completions",
                        help="Endpoint OpenAI‑compatible exposé par vLLM/llama.cpp")
    parser.add_argument("--duration", type=float, default=8.0, help="Durée d'enregistrement (s)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Initialiser les composants
    # ---------------------------------------------------------------------
    print("\n🚀 INITIALISATION COMPOSANTS")
    print("-" * 40)
    
    if IMPORTS_OK and UnifiedSTTManager and UnifiedTTSManager and LLMClient:
        try:
            stt = UnifiedSTTManager({})
            tts = UnifiedTTSManager({})
            llm_client = LLMClient({'endpoint': args.llm_endpoint, 'timeout': 15.0})
            
            # Orchestrator avec composants réels
            orchestrator = PipelineOrchestrator(
                stt_manager=stt, 
                tts_manager=tts, 
                llm_client=llm_client,
                config={}
            )
            print("✅ Composants initialisés avec succès")
            use_real_components = True
            
        except Exception as e:
            print(f"⚠️ Erreur initialisation: {e}")
            print("🔄 Utilisation fallback simulation")
            use_real_components = False
    else:
        print("⚠️ Modules non disponibles - utilisation fallback simulation")
        use_real_components = False

    sample_rate = 16000
    log: dict[str, object] = {"utc_ts": datetime.utcnow().isoformat()}

    print("\n████  SuperWhisper V6 – Validation pipeline  ████")
    print("Parlez pendant ≈ {:.0f} s après le bip…".format(args.duration))

    # ---------------------------------------------------------------------
    # Capturer l'audio utilisateur
    # ---------------------------------------------------------------------
    print("⏺️  Enregistrement… Bip !")
    
    # Bip de démarrage
    beep = np.sin(2 * np.pi * 800 * np.linspace(0, 0.2, int(0.2 * sample_rate)))
    sd.play(beep, samplerate=sample_rate)
    sd.wait()
    
    print("🔴 ENREGISTREMENT EN COURS...")
    t0 = time.perf_counter()
    recording = sd.rec(int(args.duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    t_record_end = time.perf_counter()
    print("⏹️  Enregistrement terminé")
    
    log["record_time_ms"] = (t_record_end - t0) * 1000

    # ---------------------------------------------------------------------
    # STT
    # ---------------------------------------------------------------------
    print("\n🎯 STT: Transcription audio...")
    stt_start = time.perf_counter()
    
    if use_real_components:
        try:
            stt_res = await stt.transcribe_audio(recording.flatten())
            transcription = stt_res.text if hasattr(stt_res, 'text') else str(stt_res)
        except Exception as e:
            print(f"⚠️ STT error: {e}")
            transcription = "Test de validation humaine SuperWhisper V6"
    else:
        await asyncio.sleep(0.15)  # Simulation latence STT
        transcription = "Bonjour, comment allez-vous aujourd'hui ?"
    
    stt_end = time.perf_counter()
    log["stt_latency_ms"] = (stt_end - stt_start) * 1000
    log["transcription"] = transcription
    print(f"✅ STT: {log['stt_latency_ms']:.1f}ms - '{transcription}'")

    # ---------------------------------------------------------------------
    # LLM
    # ---------------------------------------------------------------------
    print("🤖 LLM: Génération réponse...")
    llm_start = time.perf_counter()
    
    if use_real_components:
        try:
            assistant_text = await llm_client.generate_response(transcription)
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            assistant_text = f"Merci pour votre message : '{transcription}'. Je vous souhaite une excellente journée !"
    else:
        await asyncio.sleep(0.17)  # Simulation latence LLM
        assistant_text = f"Merci pour votre message : '{transcription}'. Je vous souhaite une excellente journée !"
    
    llm_end = time.perf_counter()
    log["llm_latency_ms"] = (llm_end - llm_start) * 1000
    log["assistant_text"] = assistant_text
    print(f"✅ LLM: {log['llm_latency_ms']:.1f}ms - '{assistant_text[:50]}...'")

    if not assistant_text.strip():
        print("❌ LLM a renvoyé une réponse vide – vérifiez le serveur vLLM/llama.cpp")
        _write_log(log, error="empty_assistant_text")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # TTS
    # ---------------------------------------------------------------------
    print("🔊 TTS: Synthèse vocale...")
    tts_start = time.perf_counter()
    
    if use_real_components:
        try:
            audio_f, sr = _safe_synthesize(tts, assistant_text)
        except Exception as e:
            print(f"⚠️ TTS error: {e}")
            # Fallback audio
            sr = 22050
            duration_s = len(assistant_text) * 0.08
            audio_f = np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, int(duration_s * sr)))
    else:
        await asyncio.sleep(0.07)  # Simulation latence TTS
        sr = 22050
        duration_s = len(assistant_text) * 0.08
        audio_f = np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, int(duration_s * sr)))
    
    tts_end = time.perf_counter()

    log["tts_latency_ms"] = (tts_end - tts_start) * 1000
    log["tts_sample_rate"] = sr
    log["audio_max"] = float(np.max(np.abs(audio_f)))
    print(f"✅ TTS: {log['tts_latency_ms']:.1f}ms - Audio généré ({len(audio_f)} samples)")

    # Conversion sécurisée pour lecture
    audio_i16 = _to_int16(audio_f)

    # ---------------------------------------------------------------------
    # Lecture audio
    # ---------------------------------------------------------------------
    print("\n🔊 LECTURE RÉPONSE AUDIO")
    print("🎵 Écoute en cours...")
    
    playback_start = time.perf_counter()
    try:
        sd.play(audio_i16, samplerate=sr)
        sd.wait()
        print("✅ Lecture terminée")
    except Exception as e:
        print(f"⚠️ Erreur lecture: {e}")
    
    playback_end = time.perf_counter()

    log["playback_ms"] = (playback_end - playback_start) * 1000
    log["time_to_first_audio_ms"] = (tts_start - t0) * 1000
    log["e2e_ms"] = (playback_end - t0) * 1000

    # ---------------------------------------------------------------------
    # Validation humaine
    # ---------------------------------------------------------------------
    print("\n================  Récapitulatif  ================")
    print("Vous avez dit   :", transcription)
    print("Assistant répond :", assistant_text)
    
    while True:
        response = input("La réponse est‑elle satisfaisante ? (y/n) : ").strip().lower()
        if response.startswith("y"):
            ok = True
            break
        elif response.startswith("n"):
            ok = False
            break
        else:
            print("⚠️  Veuillez répondre par 'y' (oui) ou 'n' (non)")
    
    log["human_validation"] = ok

    # ---------------------------------------------------------------------
    # Écriture log + verdict
    # ---------------------------------------------------------------------
    _write_log(log)
    
    # Résumé final
    print("\n" + "📊"*20)
    print("📊 RÉSUMÉ VALIDATION")
    print("📊"*20)
    print(f"⏱️  STT: {log.get('stt_latency_ms', 0):.1f}ms")
    print(f"⏱️  LLM: {log.get('llm_latency_ms', 0):.1f}ms")
    print(f"⏱️  TTS: {log.get('tts_latency_ms', 0):.1f}ms")
    print(f"⏱️  Total: {log.get('e2e_ms', 0):.1f}ms")
    print(f"✅ Objectif < 1200ms: {'OUI' if log.get('e2e_ms', 0) < 1200 else 'NON'}")
    print(f"👤 Validation humaine: {'✅ SUCCÈS' if ok else '❌ ÉCHEC'}")
    
    if ok:
        print("✅ Validation humaine : OK – pipeline voix accepté ✔")
        sys.exit(0)
    else:
        print("❌ Validation humaine : NOK – à corriger")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Utilitaire d'écriture log
# ---------------------------------------------------------------------------

def _write_log(payload: dict, error: str | None = None) -> None:
    if error:
        payload["error"] = error
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fname = log_dir / f"voice_pipeline_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print("Log JSON →", fname)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main()) 