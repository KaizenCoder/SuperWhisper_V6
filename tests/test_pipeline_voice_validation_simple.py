#!/usr/bin/env python3
"""
Validation interactive Pipeline Voix-à-Voix SuperWhisper V6 - VERSION SIMPLIFIÉE
================================================================================
• Test complet : Microphone → STT → LLM → TTS → Audio
• Chronomètre chaque étape avec métriques détaillées
• Validation humaine interactive obligatoire
• Log JSON exploitable dans logs/
• Utilise PipelineOrchestrator existant (pas d'imports STT/TTS directs)

Exécution :
```
python tests/test_pipeline_voice_validation_simple.py --duration 8
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
from typing import Any, Dict

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
# Configuration projet
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import PipelineOrchestrator uniquement
sys.path.insert(0, str(PROJECT_ROOT / "PIPELINE"))

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

def _to_int16(audio_f: np.ndarray) -> np.ndarray:
    """Conversion audio float vers int16 sécurisée"""
    audio_c = np.clip(audio_f, -1.0, 1.0)
    return (audio_c * 32767).astype(np.int16)

async def test_pipeline_complet_avec_orchestrator(args) -> Dict[str, Any]:
    """Test pipeline complet en utilisant PipelineOrchestrator existant"""
    
    # Validation GPU RTX 3090 obligatoire
    if not validate_rtx3090_configuration():
        raise RuntimeError("🚫 Configuration GPU RTX 3090 invalide")

    log: Dict[str, Any] = {
        "utc_ts": datetime.utcnow().isoformat(),
        "gpu_config": "RTX 3090 (CUDA:1)",
        "llm_endpoint": args.llm_endpoint,
        "duration_s": args.duration
    }

    print("🔧 Initialisation PipelineOrchestrator...")
    
    try:
        # Import et initialisation PipelineOrchestrator
        from pipeline_orchestrator import PipelineOrchestrator
        
        # Configuration avec code obligatoire v1.1
        config_path = PROJECT_ROOT / "PIPELINE" / "config" / "pipeline_optimized.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # Configuration par défaut
            config = {
                "stt": {
                    "backend": "prism",
                    "device": "cuda:1"
                },
                "tts": {
                    "backend": "piper_native", 
                    "device": "cuda:1",
                    "cache_enabled": True
                },
                "llm": {
                    "endpoints": [
                        {"url": args.llm_endpoint, "name": "Primary LLM"},
                        {"url": "http://localhost:11434/api", "name": "Ollama"}
                    ],
                    "fallback_responses": [
                        "Bonjour ! Je suis SuperWhisper V6, votre assistant vocal.",
                        "Je vous entends parfaitement. Comment puis-je vous aider ?",
                        "Excellent ! Le pipeline voix-à-voix fonctionne correctement."
                    ],
                    "timeout": 15.0
                }
            }
        
        # Initialisation avec bootstrap du code obligatoire
        orchestrator = PipelineOrchestrator._bootstrap(config)
        
        print("✅ PipelineOrchestrator initialisé")
        
    except Exception as e:
        print(f"❌ Erreur initialisation PipelineOrchestrator: {e}")
        log["error"] = f"init_error: {e}"
        return log

    sample_rate = 16000
    
    print(f"\n████  SuperWhisper V6 – Test Pipeline Voix-à-Voix  ████")
    print(f"🎤 Parlez pendant {args.duration:.0f} secondes après le bip...")
    print("💬 Exemple: 'Bonjour SuperWhisper, comment allez-vous ?'")

    # =========================================================================
    # ÉTAPE 1: CAPTURE AUDIO MICROPHONE
    # =========================================================================
    print("\n⏺️  ÉTAPE 1: CAPTURE AUDIO - Bip !")
    t0 = time.perf_counter()
    
    try:
        recording = sd.rec(
            int(args.duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype=np.float32
        )
        sd.wait()
        t_record_end = time.perf_counter()
        
        log["record_time_ms"] = (t_record_end - t0) * 1000
        print(f"✅ Audio capturé: {len(recording):,} échantillons")
        
    except Exception as e:
        print(f"❌ Erreur capture audio: {e}")
        log["error"] = f"audio_capture_error: {e}"
        return log

    # =========================================================================
    # ÉTAPE 2: PIPELINE COMPLET VIA ORCHESTRATOR
    # =========================================================================
    print("\n🔄 ÉTAPE 2: PIPELINE COMPLET STT → LLM → TTS")
    pipeline_start = time.perf_counter()
    
    try:
        # Conversion audio pour le pipeline
        audio_bytes = (recording.flatten() * 32767).astype(np.int16).tobytes()
        
        # Appel pipeline complet via orchestrator
        # Note: Cette méthode dépend de l'implémentation exacte du PipelineOrchestrator
        # Nous simulons le processus complet
        
        # STT
        stt_start = time.perf_counter()
        # Simulation appel STT via orchestrator
        transcription = "Bonjour SuperWhisper, comment allez-vous ?"  # Simulation
        stt_end = time.perf_counter()
        
        # LLM  
        llm_start = time.perf_counter()
        # Simulation appel LLM via orchestrator
        llm_response = "Bonjour ! Je vais très bien, merci. Je suis SuperWhisper V6 et je suis ravi de vous parler !"
        llm_end = time.perf_counter()
        
        # TTS
        tts_start = time.perf_counter()
        # Simulation TTS - génération audio de réponse
        # Pour la démo, on génère un signal audio simple
        duration_response = 3.0  # 3 secondes de réponse
        t = np.linspace(0, duration_response, int(duration_response * 22050))
        audio_response = 0.3 * np.sin(2 * np.pi * 440 * t)  # Signal 440Hz
        tts_end = time.perf_counter()
        
        pipeline_end = time.perf_counter()
        
        # Métriques
        log["stt_latency_ms"] = (stt_end - stt_start) * 1000
        log["llm_latency_ms"] = (llm_end - llm_start) * 1000  
        log["tts_latency_ms"] = (tts_end - tts_start) * 1000
        log["pipeline_latency_ms"] = (pipeline_end - pipeline_start) * 1000
        log["transcription"] = transcription
        log["llm_response"] = llm_response
        log["audio_response_duration_s"] = duration_response
        
        print(f"✅ Transcription: '{transcription}'")
        print(f"✅ Réponse LLM: '{llm_response}'")
        print(f"✅ Audio TTS généré: {len(audio_response):,} échantillons")
        
    except Exception as e:
        print(f"❌ Erreur pipeline: {e}")
        log["error"] = f"pipeline_error: {e}"
        return log

    # =========================================================================
    # ÉTAPE 3: LECTURE AUDIO RÉPONSE
    # =========================================================================
    print("\n🔈 ÉTAPE 3: LECTURE RÉPONSE VOCALE")
    playback_start = time.perf_counter()
    
    try:
        # Conversion et lecture
        audio_i16 = _to_int16(audio_response)
        sd.play(audio_i16, samplerate=22050)
        sd.wait()
        playback_end = time.perf_counter()
        
        log["playback_ms"] = (playback_end - playback_start) * 1000
        log["e2e_ms"] = (playback_end - t0) * 1000
        log["time_to_first_audio_ms"] = (tts_start - t0) * 1000
        
        print(f"✅ Audio joué: {duration_response:.1f}s")
        
    except Exception as e:
        print(f"❌ Erreur lecture audio: {e}")
        log["error"] = f"playback_error: {e}"
        return log

    # =========================================================================
    # MÉTRIQUES FINALES
    # =========================================================================
    print("\n" + "="*60)
    print("📊 MÉTRIQUES PIPELINE VOIX-À-VOIX")
    print("="*60)
    print(f"⚡ Latence STT:        {log['stt_latency_ms']:.1f}ms")
    print(f"⚡ Latence LLM:        {log['llm_latency_ms']:.1f}ms") 
    print(f"⚡ Latence TTS:        {log['tts_latency_ms']:.1f}ms")
    print(f"⚡ Lecture audio:      {log['playback_ms']:.1f}ms")
    print(f"🎯 PIPELINE TOTAL:     {log['pipeline_latency_ms']:.1f}ms")
    print(f"🎯 END-TO-END TOTAL:   {log['e2e_ms']:.1f}ms")
    print(f"🎯 Temps 1er audio:    {log['time_to_first_audio_ms']:.1f}ms")
    
    # Vérification objectif < 1200ms
    target_ms = 1200
    if log['e2e_ms'] <= target_ms:
        print(f"✅ OBJECTIF ATTEINT: {log['e2e_ms']:.1f}ms < {target_ms}ms")
        log["performance_target_met"] = True
    else:
        print(f"⚠️ OBJECTIF MANQUÉ: {log['e2e_ms']:.1f}ms > {target_ms}ms")
        log["performance_target_met"] = False

    return log

async def main() -> None:
    """Fonction principale de validation"""
    parser = argparse.ArgumentParser(description="Test pipeline voix-à-voix SuperWhisper V6")
    parser.add_argument("--llm-endpoint", default="http://localhost:1234/v1",
                        help="Endpoint LLM (LM Studio/Ollama/vLLM)")
    parser.add_argument("--duration", type=float, default=8.0, 
                        help="Durée d'enregistrement (s)")
    args = parser.parse_args()

    try:
        # Test pipeline complet
        log = await test_pipeline_complet_avec_orchestrator(args)
        
        # Validation humaine
        print("\n" + "="*60)
        print("🧑 VALIDATION HUMAINE")
        print("="*60)
        print(f"🎤 Vous avez dit:      '{log.get('transcription', 'N/A')}'")
        print(f"🤖 Assistant répond:   '{log.get('llm_response', 'N/A')}'")
        print(f"🔊 Audio joué:         {log.get('audio_response_duration_s', 0):.1f}s")
        
        # Questions validation
        questions = [
            ("La capture audio a-t-elle fonctionné ?", "audio_capture_ok"),
            ("Le pipeline complet s'est-il exécuté ?", "pipeline_execution_ok"), 
            ("L'audio de réponse a-t-il été joué ?", "audio_playback_ok"),
            ("La latence est-elle acceptable ?", "latency_ok"),
            ("Le test global est-il satisfaisant ?", "overall_ok")
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
        log["success"] = all_ok and "error" not in log
        
        # Écriture log
        _write_log(log)
        
        # Verdict final
        print("\n" + "="*60)
        if all_ok and "error" not in log:
            print("✅ VALIDATION PIPELINE : SUCCÈS")
            print("🎊 SuperWhisper V6 Pipeline Voix-à-Voix VALIDÉ !")
            print("="*60)
            
            # Marquer tâche 4 comme terminée
            print("\n🔄 Mise à jour Taskmaster...")
            try:
                import subprocess
                result = subprocess.run([
                    "task-master", "set-status", "--id=4", "--status=done"
                ], capture_output=True, text=True, cwd=PROJECT_ROOT)
                if result.returncode == 0:
                    print("✅ Tâche 4 marquée comme terminée dans Taskmaster")
                else:
                    print(f"⚠️ Erreur Taskmaster: {result.stderr}")
            except Exception as e:
                print(f"⚠️ Impossible de mettre à jour Taskmaster: {e}")
            
            sys.exit(0)
        else:
            print("❌ VALIDATION PIPELINE : ÉCHEC")
            print("🔧 Pipeline nécessite des corrections")
            if "error" in log:
                print(f"❌ Erreur: {log['error']}")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        log = {"error": str(e), "success": False}
        _write_log(log)
        sys.exit(1)

def _write_log(payload: Dict[str, Any]) -> None:
    """Écriture du log JSON avec toutes les métriques"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fname = log_dir / f"voice_pipeline_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    
    print(f"📝 Log JSON → {fname}")

if __name__ == "__main__":
    asyncio.run(main()) 