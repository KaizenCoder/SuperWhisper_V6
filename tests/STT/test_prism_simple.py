#!/usr/bin/env python3
"""
Test simple intégration faster-whisper - RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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

import torch
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

def validate_rtx3090_mandatory():
    """Validation obligatoire RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

@dataclass
class STTResult:
    """Résultat de transcription STT"""
    text: str
    confidence: float
    processing_time: float
    rtf: float  # Real-Time Factor
    model_used: str
    audio_duration: float
    error: Optional[str] = None

def test_faster_whisper_import():
    """Test import faster-whisper"""
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper importé avec succès")
        return True
    except ImportError as e:
        print(f"❌ Erreur import faster-whisper: {e}")
        return False

def test_faster_whisper_model():
    """Test chargement modèle faster-whisper"""
    try:
        from faster_whisper import WhisperModel
        
        print("🔄 Chargement modèle tiny...")
        start_time = time.time()
        
        model = WhisperModel(
            "tiny",
            device="cuda",
            compute_type="float16"
        )
        
        load_time = time.time() - start_time
        print(f"✅ Modèle tiny chargé en {load_time:.2f}s")
        
        # Test GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"📊 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        return model
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return None

def generate_test_audio(duration=3.0, sample_rate=16000):
    """Génère un audio de test synthétique"""
    # Génération d'un signal sinusoïdal simple
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # La note A4
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"🎵 Audio test généré: {duration}s, {sample_rate}Hz, {len(audio)} samples")
    return audio

def test_transcription(model, audio, sample_rate=16000):
    """Test transcription avec faster-whisper"""
    try:
        print("🎤 Début transcription...")
        start_time = time.time()
        
        # Transcription
        segments, info = model.transcribe(
            audio,
            beam_size=1,
            language="fr",
            condition_on_previous_text=False
        )
        
        # Collecte des segments
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text)
        
        processing_time = time.time() - start_time
        audio_duration = len(audio) / sample_rate
        rtf = processing_time / audio_duration
        
        result = STTResult(
            text=" ".join(text_segments).strip(),
            confidence=info.language_probability,
            processing_time=processing_time,
            rtf=rtf,
            model_used="tiny",
            audio_duration=audio_duration
        )
        
        print(f"✅ Transcription terminée:")
        print(f"   📝 Texte: '{result.text}'")
        print(f"   ⏱️ Temps: {result.processing_time:.3f}s")
        print(f"   🎯 RTF: {result.rtf:.3f}")
        print(f"   🔊 Durée audio: {result.audio_duration:.1f}s")
        print(f"   📊 Confiance: {result.confidence:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        return STTResult(
            text="",
            confidence=0.0,
            processing_time=0.0,
            rtf=0.0,
            model_used="tiny",
            audio_duration=0.0,
            error=str(e)
        )

def main():
    """Test principal"""
    print("🚀 Démarrage test intégration faster-whisper RTX 3090")
    print("=" * 60)
    
    try:
        # 1. Validation RTX 3090
        print("\n1️⃣ Validation RTX 3090...")
        validate_rtx3090_mandatory()
        
        # 2. Test import
        print("\n2️⃣ Test import faster-whisper...")
        if not test_faster_whisper_import():
            return False
        
        # 3. Test chargement modèle
        print("\n3️⃣ Test chargement modèle...")
        model = test_faster_whisper_model()
        if model is None:
            return False
        
        # 4. Test transcription
        print("\n4️⃣ Test transcription...")
        audio = generate_test_audio(duration=3.0)
        result = test_transcription(model, audio)
        
        if result.error:
            print(f"❌ Erreur transcription: {result.error}")
            return False
        
        # 5. Validation performance
        print("\n5️⃣ Validation performance...")
        if result.rtf < 0.5:  # Objectif RTF < 0.5 pour tiny
            print(f"✅ Performance RTF excellente: {result.rtf:.3f} < 0.5")
        else:
            print(f"⚠️ Performance RTF acceptable: {result.rtf:.3f}")
        
        print("\n🎉 Test intégration faster-whisper RÉUSSI!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 