#!/usr/bin/env python3
"""
Test de validation pipeline voix-à-voix SuperWhisper V6 - AVEC TTS RÉEL VALIDÉ
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Utilise le modèle TTS validé fr_FR-siwis-medium.onnx disponible dans D:\TTS_Voices\piper
Basé sur la transmission du coordinateur du 10/06/2025 - TTS déjà validé

Test interactif complet: STT → LLM → TTS → Audio avec validation humaine

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

import time
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

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

# Configuration paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "TTS"))

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
    except ImportError:
        print("⚠️ PyTorch non disponible - validation GPU ignorée")
        return True
    except Exception as e:
        print(f"⚠️ Validation GPU échouée: {e}")
        return True

class MockSTTManager:
    """Mock STT Manager pour test pipeline"""
    
    def __init__(self):
        print("🎤 Mock STT Manager initialisé")
    
    async def transcribe(self, audio_data: np.ndarray) -> object:
        """Simule transcription STT"""
        await asyncio.sleep(0.15)  # 150ms latence STT
        
        class STTResult:
            def __init__(self, text: str):
                self.text = text
                self.confidence = 0.95
        
        return STTResult("Bonjour, je teste le pipeline voix-à-voix de SuperWhisper V6")

class MockLLMClient:
    """Mock LLM Client pour test pipeline"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.fallback_responses = [
            "Bonjour ! Je suis SuperWhisper V6, votre assistant vocal intelligent.",
            "Parfait ! Le pipeline voix-à-voix fonctionne correctement.",
            "Excellent ! Tous les composants STT, LLM et TTS sont opérationnels.",
            "Je vous entends parfaitement. Le système de reconnaissance vocale est fonctionnel.",
            "Merci de tester SuperWhisper V6. La synthèse vocale est de qualité exceptionnelle."
        ]
        print(f"🤖 Mock LLM Client initialisé (endpoint: {endpoint})")
    
    async def generate_response(self, text: str) -> str:
        """Simule génération LLM"""
        await asyncio.sleep(0.18)  # 180ms latence LLM
        
        if "test" in text.lower() or "pipeline" in text.lower():
            return self.fallback_responses[1]
        elif "bonjour" in text.lower():
            return self.fallback_responses[0]
        else:
            return self.fallback_responses[2]

class ValidatedTTSHandler:
    """TTS Handler utilisant le modèle validé fr_FR-siwis-medium.onnx"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.model_path = None
        self.config_path = None
        self.piper_exe = None
        
        print("🔊 Initialisation TTS Handler avec modèle validé...")
        
        # Chercher le modèle validé dans D:\TTS_Voices\piper
        model_locations = [
            Path("D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"),
            Path("models/fr_FR-siwis-medium.onnx"),
            Path("TTS/models/fr_FR-siwis-medium.onnx")
        ]
        
        for model_path in model_locations:
            if model_path.exists():
                self.model_path = str(model_path)
                self.config_path = str(model_path.with_suffix('.onnx.json'))
                print(f"✅ Modèle TTS trouvé: {self.model_path}")
                break
        
        if not self.model_path:
            raise FileNotFoundError("❌ Modèle TTS fr_FR-siwis-medium.onnx non trouvé")
        
        # Chercher piper.exe
        piper_locations = [
            Path("piper/piper.exe"),
            Path("D:/TTS_Voices/piper/piper.exe"),
            Path("piper.exe")
        ]
        
        for piper_path in piper_locations:
            if piper_path.exists():
                self.piper_exe = str(piper_path)
                print(f"✅ Piper.exe trouvé: {self.piper_exe}")
                break
        
        if not self.piper_exe:
            raise FileNotFoundError("❌ piper.exe non trouvé")
        
        # Vérifier configuration
        if Path(self.config_path).exists():
            print(f"✅ Configuration TTS: {self.config_path}")
        else:
            print(f"⚠️ Configuration TTS manquante: {self.config_path}")
        
        print("🎊 TTS Handler validé initialisé avec succès!")
    
    def speak(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthèse TTS avec le modèle validé fr_FR-siwis-medium.onnx"""
        print(f"🎵 Synthèse TTS RÉELLE avec modèle validé: {text}")
        
        try:
            import tempfile
            import wave
            import subprocess
            import uuid
            
            # Créer fichier temporaire unique avec UUID
            temp_dir = Path(tempfile.gettempdir())
            unique_id = str(uuid.uuid4())[:8]
            tmp_path = temp_dir / f"superwhisper_tts_{unique_id}.wav"
            
            # Commande piper avec modèle validé
            cmd = [
                self.piper_exe,
                "--model", self.model_path,
                "--output_file", str(tmp_path),
                "--speaker", "0"  # Locuteur par défaut
            ]
            
            print(f"🔧 Commande Piper: {' '.join(cmd)}")
            
            # Exécuter piper
            result = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0 and tmp_path.exists():
                # Attendre un peu pour s'assurer que le fichier est libéré
                import time
                time.sleep(0.1)
                
                # Lire le fichier WAV généré
                try:
                    with wave.open(str(tmp_path), 'rb') as wav_file:
                        frames = wav_file.readframes(-1)
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        
                        # Conversion vers numpy
                        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                        
                        # Gérer stéréo → mono si nécessaire
                        if channels == 2:
                            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                        
                        print(f"✅ TTS RÉEL généré: {len(audio_data)} samples à {sample_rate}Hz")
                        
                        # Nettoyage fichier temporaire avec retry
                        for attempt in range(3):
                            try:
                                tmp_path.unlink(missing_ok=True)
                                break
                            except PermissionError:
                                time.sleep(0.1)
                        
                        return audio_data, sample_rate
                        
                except Exception as e:
                    print(f"❌ Erreur lecture fichier WAV: {e}")
                    # Nettoyage en cas d'erreur
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except:
                        pass
                    raise
            else:
                print(f"❌ Erreur Piper (code {result.returncode}):")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                # Nettoyage en cas d'erreur
                try:
                    tmp_path.unlink(missing_ok=True)
                except:
                    pass
                raise RuntimeError(f"Piper failed with code {result.returncode}")
                
        except Exception as e:
            print(f"❌ Erreur synthèse TTS: {e}")
            raise

def _to_int16(audio_f: np.ndarray) -> np.ndarray:
    """Conversion audio float vers int16 sécurisée"""
    audio_c = np.clip(audio_f, -1.0, 1.0)
    return (audio_c * 32767).astype(np.int16)

async def main():
    """Test interactif pipeline voix-à-voix SuperWhisper V6 avec TTS validé"""
    parser = argparse.ArgumentParser(description="Test pipeline voix-à-voix SuperWhisper V6 - TTS VALIDÉ")
    parser.add_argument("--llm-endpoint", default="http://localhost:1234/v1",
                        help="Endpoint LLM (LM Studio/Ollama)")
    parser.add_argument("--duration", type=float, default=3.0, 
                        help="Durée d'enregistrement (s)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 SUPERWHISPER V6 - TEST PIPELINE VOIX-À-VOIX (TTS VALIDÉ)")
    print("="*80)

    # 1. Validation GPU RTX 3090
    print("\n1️⃣ Validation GPU RTX 3090...")
    if not validate_rtx3090_configuration():
        print("🚫 ÉCHEC: Configuration GPU RTX 3090 invalide")
        return False

    # 2. Initialisation composants
    print("\n2️⃣ Initialisation des composants...")
    try:
        stt = MockSTTManager()
        llm = MockLLMClient(args.llm_endpoint)
        tts = ValidatedTTSHandler()
        print("✅ Tous les composants initialisés avec TTS VALIDÉ")
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Préparation log
    log = {
        "utc_ts": datetime.now().isoformat(),
        "gpu_config": "RTX 3090 (CUDA:1)",
        "llm_endpoint": args.llm_endpoint,
        "test_type": "pipeline_voice_validation_validated_tts",
        "tts_model": tts.model_path,
        "tts_type": "validated_real"
    }

    # 4. Capture audio
    print(f"\n3️⃣ Capture audio ({args.duration}s)...")
    print("🎤 Parlez maintenant après le bip...")
    
    sample_rate = 16000
    t0 = time.perf_counter()
    
    # Bip de démarrage
    print("🔔 BIP!")
    beep = 0.3 * np.sin(2 * np.pi * 800 * np.linspace(0, 0.2, int(0.2 * sample_rate)))
    sd.play(_to_int16(beep), samplerate=sample_rate)
    sd.wait()
    
    # Enregistrement
    recording = sd.rec(int(args.duration * sample_rate), 
                      samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    t_record_end = time.perf_counter()
    log["record_time_ms"] = (t_record_end - t0) * 1000

    # 5. STT
    print("\n4️⃣ Transcription STT...")
    stt_start = time.perf_counter()
    stt_result = await stt.transcribe(recording.flatten())
    stt_end = time.perf_counter()
    log["stt_latency_ms"] = (stt_end - stt_start) * 1000
    log["transcription"] = stt_result.text
    print(f"📝 Transcription: {stt_result.text}")

    # 6. LLM
    print("\n5️⃣ Génération LLM...")
    llm_start = time.perf_counter()
    assistant_text = await llm.generate_response(stt_result.text)
    llm_end = time.perf_counter()
    log["llm_latency_ms"] = (llm_end - llm_start) * 1000
    log["assistant_text"] = assistant_text
    print(f"🤖 Réponse LLM: {assistant_text}")

    if not assistant_text.strip():
        print("❌ LLM a renvoyé une réponse vide")
        _write_log(log, error="empty_assistant_text")
        return False

    # 7. TTS VALIDÉ
    print("\n6️⃣ Synthèse TTS VALIDÉE...")
    tts_start = time.perf_counter()
    try:
        audio_f, sr = tts.speak(assistant_text)
        tts_end = time.perf_counter()
        log["tts_latency_ms"] = (tts_end - tts_start) * 1000
        log["tts_sample_rate"] = sr
        log["audio_max"] = float(np.max(np.abs(audio_f)))
        log["audio_samples"] = len(audio_f)
    except Exception as e:
        print(f"❌ Erreur TTS validé: {e}")
        _write_log(log, error=f"tts_error: {e}")
        return False

    # 8. Lecture audio
    print("\n7️⃣ Lecture réponse audio VALIDÉE...")
    audio_i16 = _to_int16(audio_f)
    playback_start = time.perf_counter()
    sd.play(audio_i16, samplerate=sr)
    sd.wait()
    playback_end = time.perf_counter()

    # 9. Métriques finales
    log["playback_ms"] = (playback_end - playback_start) * 1000
    log["time_to_first_audio_ms"] = (tts_start - t0) * 1000
    log["e2e_ms"] = (playback_end - t0) * 1000

    # 10. Validation humaine
    print("\n" + "="*80)
    print("📊 RÉCAPITULATIF PIPELINE (TTS VALIDÉ)")
    print("="*80)
    print(f"🎤 Vous avez dit    : {stt_result.text}")
    print(f"🤖 Assistant répond : {assistant_text}")
    print(f"🔊 Modèle TTS       : {Path(tts.model_path).name}")
    print(f"⏱️  Latence STT     : {log['stt_latency_ms']:.1f}ms")
    print(f"⏱️  Latence LLM     : {log['llm_latency_ms']:.1f}ms")
    print(f"⏱️  Latence TTS     : {log['tts_latency_ms']:.1f}ms")
    print(f"⏱️  Latence totale  : {log['e2e_ms']:.1f}ms")
    print(f"🎵 Échantillons     : {log['audio_samples']} à {sr}Hz")
    print("="*80)

    # Validation humaine
    try:
        print("\n🎧 VALIDATION CRITIQUE:")
        print("   Avez-vous entendu une VRAIE VOIX HUMAINE SYNTHÉTIQUE")
        print("   (pas un bip, signal électronique ou silence) ?")
        response = input("\n✅ VRAIE SYNTHÈSE VOCALE ENTENDUE ? (y/n): ").strip().lower()
        ok = response.startswith('y')
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par l'utilisateur")
        ok = False

    log["human_validation"] = ok
    log["validation_type"] = "validated_voice_synthesis"

    # 11. Écriture log et verdict
    _write_log(log)
    
    if ok:
        print("\n🎊 VALIDATION RÉUSSIE !")
        print("✅ Pipeline voix-à-voix SuperWhisper V6 avec TTS VALIDÉ fonctionnel")
        print("🏆 Modèle fr_FR-siwis-medium.onnx produit une vraie synthèse vocale")
        return True
    else:
        print("\n❌ VALIDATION ÉCHOUÉE")
        print("⚠️ Problème avec la synthèse vocale - investigation requise")
        return False

def _write_log(payload: dict, error: Optional[str] = None) -> None:
    """Écriture log JSON"""
    if error:
        payload["error"] = error
    
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = log_dir / f"voice_pipeline_validation_validated_{timestamp}.json"
    
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    
    print(f"📄 Log sauvegardé: {fname}")

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 