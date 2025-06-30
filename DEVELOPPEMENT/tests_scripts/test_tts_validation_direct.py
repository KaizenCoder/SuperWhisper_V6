#!/usr/bin/env python3
"""
Validation TTS SuperWhisper V6 - Test Direct Windows SAPI
=========================================================
• Test TTS direct avec Windows SAPI (fallback validé Phase 3)
• Évite les problèmes d'imports complexes
• Configuration RTX 3090 obligatoire
• Validation humaine interactive

Exécution :
```
python tests/test_tts_validation_direct.py
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

import os
import sys
import time
import asyncio
import subprocess
import tempfile
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
        return True, torch.cuda.get_device_name(0), gpu_memory
    except Exception as e:
        print(f"⚠️ Validation GPU échouée: {e}")
        return False, str(e), 0

def test_windows_sapi_tts(text: str) -> tuple[bool, bytes, float]:
    """Test TTS direct avec Windows SAPI"""
    try:
        # Script PowerShell pour Windows SAPI
        powershell_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# Configuration voix française
$voices = $synth.GetInstalledVoices()
foreach ($voice in $voices) {{
    if ($voice.VoiceInfo.Culture.Name -eq "fr-FR") {{
        $synth.SelectVoice($voice.VoiceInfo.Name)
        break
    }}
}}

# Configuration audio
$synth.Rate = 0
$synth.Volume = 100

# Fichier temporaire WAV
$tempFile = [System.IO.Path]::GetTempFileName() + ".wav"
$synth.SetOutputToWaveFile($tempFile)

# Synthèse
$synth.Speak("{text}")

# Nettoyage
$synth.SetOutputToDefaultAudioDevice()
$synth.Dispose()

# Retourner le chemin du fichier
Write-Output $tempFile
'''
        
        start_time = time.perf_counter()
        
        # Exécution PowerShell
        result = subprocess.run(
            ["powershell", "-Command", powershell_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        if result.returncode != 0:
            raise RuntimeError(f"PowerShell SAPI échoué: {result.stderr}")
        
        # Récupération du fichier WAV
        wav_file_path = result.stdout.strip()
        
        if not Path(wav_file_path).exists():
            raise RuntimeError(f"Fichier WAV non créé: {wav_file_path}")
        
        # Lecture du fichier WAV
        with open(wav_file_path, 'rb') as f:
            wav_data = f.read()
        
        # Nettoyage
        try:
            Path(wav_file_path).unlink()
        except:
            pass
        
        return True, wav_data, latency_ms
        
    except Exception as e:
        print(f"❌ Erreur SAPI TTS: {e}")
        return False, b'', 0

async def test_tts_direct():
    """Test TTS direct avec validation humaine"""
    
    print("\n████  SuperWhisper V6 – Test TTS Direct Windows SAPI  ████")
    
    # =========================================================================
    # ÉTAPE 1: VALIDATION GPU RTX 3090
    # =========================================================================
    print("\n🔧 ÉTAPE 1: VALIDATION GPU RTX 3090")
    gpu_ok, gpu_name, gpu_memory = validate_rtx3090_configuration()
    
    if not gpu_ok:
        print("❌ Validation GPU échouée - Arrêt du test")
        return False

    # =========================================================================
    # ÉTAPE 2: TEST TTS WINDOWS SAPI
    # =========================================================================
    print("\n🔊 ÉTAPE 2: TEST TTS WINDOWS SAPI")
    
    test_text = "Bonjour ! Je suis SuperWhisper V6, votre assistant vocal. Test de synthèse vocale avec Windows SAPI."
    print(f"📝 Texte à synthétiser: '{test_text}'")
    
    print("🔄 Synthèse TTS en cours...")
    success, wav_data, latency_ms = test_windows_sapi_tts(test_text)
    
    if not success:
        print("❌ Échec synthèse TTS")
        return False
    
    print(f"✅ Synthèse TTS réussie: {len(wav_data):,} bytes")
    print(f"⚡ Latence TTS: {latency_ms:.1f}ms")
    
    # Validation performance
    if latency_ms < 100:
        print(f"🎯 Performance EXCELLENTE (< 100ms)")
    elif latency_ms < 1000:
        print(f"✅ Performance BONNE (< 1s)")
    elif latency_ms < 3000:
        print(f"⚠️ Performance ACCEPTABLE (< 3s)")
    else:
        print(f"❌ Performance DÉGRADÉE (> 3s)")

    # =========================================================================
    # ÉTAPE 3: LECTURE AUDIO
    # =========================================================================
    print("\n🔈 ÉTAPE 3: LECTURE AUDIO")
    
    try:
        print("🔄 Conversion WAV vers numpy...")
        
        # Conversion WAV vers numpy pour lecture
        import wave
        import io
        
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(-1)
            audio_np = np.frombuffer(frames, dtype=np.int16)
        
        duration_s = len(audio_np) / sample_rate
        print(f"🎵 Sample rate: {sample_rate}Hz")
        print(f"⏱️ Durée audio: {duration_s:.1f}s")
        print(f"📊 Échantillons: {len(audio_np):,}")
        
        print("🔊 Lecture audio - Écoutez attentivement...")
        sd.play(audio_np, samplerate=sample_rate)
        sd.wait()
        
        print("✅ Audio joué avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lecture audio: {e}")
        return False

    # =========================================================================
    # VALIDATION HUMAINE
    # =========================================================================
    print("\n" + "="*60)
    print("🧑 VALIDATION HUMAINE TTS WINDOWS SAPI")
    print("="*60)
    print(f"📝 Texte synthétisé: '{test_text}'")
    print(f"⚡ Latence TTS: {latency_ms:.1f}ms")
    print(f"🎵 Durée audio: {duration_s:.1f}s")
    print(f"🎮 GPU utilisée: {gpu_name}")
    print(f"🔧 Handler: Windows SAPI (Fallback Phase 3)")
    print(f"🏆 Phase 3 TTS: 4 backends validés (PiperNative, PiperCli, SAPI, Silent)")
    
    # Questions validation
    questions = [
        "Avez-vous entendu l'audio TTS ?",
        "La voix française est-elle claire et compréhensible ?", 
        "La latence TTS est-elle acceptable pour une conversation ?",
        "Le TTS Windows SAPI fonctionne-t-il correctement ?"
    ]
    
    all_ok = True
    for question in questions:
        response = input(f"❓ {question} (y/n): ").strip().lower()
        if not response.startswith("y"):
            all_ok = False
    
    # Verdict final
    print("\n" + "="*60)
    if all_ok:
        print("✅ VALIDATION TTS WINDOWS SAPI : SUCCÈS")
        print("🎊 TTS SuperWhisper V6 VALIDÉ !")
        print("🔧 Fallback SAPI fonctionnel (Phase 3)")
        print("="*60)
        return True
    else:
        print("❌ VALIDATION TTS WINDOWS SAPI : ÉCHEC")
        print("🔧 TTS nécessite des corrections")
        print("="*60)
        return False

async def main():
    """Point d'entrée principal"""
    try:
        success = await test_tts_direct()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 