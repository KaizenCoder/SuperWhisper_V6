#!/usr/bin/env python3
"""
Diagnostic STT Simple - SuperWhisper V6 Phase 4
🔧 DIAGNOSTIC: Identifier le problème VAD avec méthode synchrone

Mission: Identifier précisément où se bloque la transcription
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🔧 Diagnostic STT Simple - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def validate_rtx3090():
    """Validation RTX 3090"""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

def test_audio_court():
    """Test avec audio court synthétique"""
    
    print("\n" + "="*50)
    print("🔧 TEST 1 - AUDIO COURT SYNTHÉTIQUE")
    print("="*50)
    
    validate_rtx3090()
    
    # Import faster-whisper direct
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper importé")
    except ImportError:
        print("❌ faster-whisper non disponible")
        return False
    
    # Audio synthétique court (3 secondes)
    SAMPLE_RATE = 16000
    duration = 3.0
    samples = int(duration * SAMPLE_RATE)
    
    # Générer audio simple
    t = np.linspace(0, duration, samples)
    frequency = 440  # La 440Hz
    audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"📊 Audio généré: {len(audio)} samples, {duration}s")
    
    # Test faster-whisper direct
    print(f"🚀 Initialisation faster-whisper...")
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")
    print(f"✅ Modèle chargé")
    
    print(f"🤖 Transcription synchrone...")
    start_time = time.time()
    
    try:
        # Transcription SYNCHRONE avec paramètres VAD corrigés
        vad_parameters = {
            "threshold": 0.3,                    
            "min_speech_duration_ms": 100,       
            "max_speech_duration_s": 60,         
            "min_silence_duration_ms": 1000,     
            "speech_pad_ms": 400                 
        }
        
        segments, info = model.transcribe(
            audio,
            language="fr",
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=vad_parameters
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Collecter résultats
        segments_list = list(segments)
        text = " ".join([seg.text for seg in segments_list])
        
        print(f"✅ Transcription réussie en {duration_ms:.0f}ms")
        print(f"📝 Texte: '{text}'")
        print(f"📊 Segments: {len(segments_list)}")
        print(f"⚡ RTF: {(duration_ms/1000)/duration:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        return False

def test_audio_long():
    """Test avec audio long synthétique"""
    
    print("\n" + "="*50)
    print("🔧 TEST 2 - AUDIO LONG SYNTHÉTIQUE")
    print("="*50)
    
    # Import faster-whisper direct
    from faster_whisper import WhisperModel
    
    # Audio synthétique long (30 secondes)
    SAMPLE_RATE = 16000
    duration = 30.0
    samples = int(duration * SAMPLE_RATE)
    
    # Générer audio varié pour simuler parole
    t = np.linspace(0, duration, samples)
    
    # Mélange de fréquences pour simuler parole
    audio = np.zeros(samples, dtype=np.float32)
    
    # Ajouter plusieurs segments avec pauses
    segment_duration = 3.0  # 3s par segment
    pause_duration = 1.0    # 1s de pause
    
    for i in range(6):  # 6 segments
        start_seg = int(i * (segment_duration + pause_duration) * SAMPLE_RATE)
        end_seg = int(start_seg + segment_duration * SAMPLE_RATE)
        
        if end_seg < len(audio):
            # Fréquence variable par segment
            freq = 440 + i * 50
            t_seg = np.linspace(0, segment_duration, end_seg - start_seg)
            audio[start_seg:end_seg] = 0.3 * np.sin(2 * np.pi * freq * t_seg)
    
    print(f"📊 Audio long généré: {len(audio)} samples, {duration}s")
    
    # Réutiliser le modèle (déjà chargé)
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")
    
    print(f"🤖 Transcription audio long avec VAD corrigé...")
    start_time = time.time()
    
    try:
        # Transcription avec paramètres VAD corrigés
        vad_parameters = {
            "threshold": 0.3,                    
            "min_speech_duration_ms": 100,       
            "max_speech_duration_s": 60,         
            "min_silence_duration_ms": 1000,     
            "speech_pad_ms": 400                 
        }
        
        segments, info = model.transcribe(
            audio,
            language="fr",
            beam_size=5,
            vad_filter=True,
            vad_parameters=vad_parameters
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Collecter résultats
        segments_list = list(segments)
        text = " ".join([seg.text for seg in segments_list])
        
        print(f"✅ Transcription longue réussie en {duration_ms:.0f}ms")
        print(f"📝 Texte: '{text}'")
        print(f"📊 Segments détectés: {len(segments_list)}")
        print(f"⚡ RTF: {(duration_ms/1000)/duration:.3f}")
        
        # Analyser segments VAD
        if segments_list:
            print(f"\n🔍 ANALYSE SEGMENTS VAD:")
            for i, seg in enumerate(segments_list[:5]):  # Max 5 premiers
                print(f"   Seg {i+1}: {seg.start:.1f}s → {seg.end:.1f}s ({seg.end-seg.start:.1f}s)")
                print(f"          Texte: '{seg.text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur transcription longue: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vad_validation():
    """Test final de validation VAD"""
    
    print("\n" + "="*50)
    print("🎯 TEST 3 - VALIDATION CORRECTION VAD")
    print("="*50)
    
    from faster_whisper import WhisperModel
    
    # Créer un audio test de 30s avec beaucoup de segments
    SAMPLE_RATE = 16000
    duration = 30.0
    samples = int(duration * SAMPLE_RATE)
    
    audio = np.zeros(samples, dtype=np.float32)
    
    # 15 segments de 1s avec 1s de pause entre (simule phrase longue)
    for i in range(15):
        start_seg = int(i * 2 * SAMPLE_RATE)  # Tous les 2s
        end_seg = int(start_seg + 1 * SAMPLE_RATE)  # 1s de durée
        
        if end_seg < len(audio):
            # Fréquence différente par "mot"
            freq = 200 + i * 30  # Variation fréquence
            t_seg = np.linspace(0, 1.0, end_seg - start_seg)
            audio[start_seg:end_seg] = 0.2 * np.sin(2 * np.pi * freq * t_seg)
    
    print(f"📊 Audio test validation: 15 segments sur {duration}s")
    
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")
    
    # Test 1: Paramètres VAD par défaut (problématiques)
    print(f"\n🧪 Test VAD défaut (problématique):")
    try:
        start_time = time.time()
        segments, info = model.transcribe(
            audio,
            language="fr",
            beam_size=5,
            vad_filter=True
            # Pas de vad_parameters = utilise défaut
        )
        
        duration_ms = (time.time() - start_time) * 1000
        segments_list = list(segments)
        
        print(f"   ⏱️ Durée: {duration_ms:.0f}ms")
        print(f"   📊 Segments détectés: {len(segments_list)}")
        print(f"   ⚡ RTF: {(duration_ms/1000)/duration:.3f}")
        
    except Exception as e:
        print(f"   ❌ Erreur VAD défaut: {e}")
    
    # Test 2: Paramètres VAD corrigés (solution)
    print(f"\n🔧 Test VAD corrigé (solution):")
    try:
        vad_parameters = {
            "threshold": 0.3,                    
            "min_speech_duration_ms": 100,       
            "max_speech_duration_s": 60,         
            "min_silence_duration_ms": 1000,     
            "speech_pad_ms": 400                 
        }
        
        start_time = time.time()
        segments, info = model.transcribe(
            audio,
            language="fr",
            beam_size=5,
            vad_filter=True,
            vad_parameters=vad_parameters
        )
        
        duration_ms = (time.time() - start_time) * 1000
        segments_list = list(segments)
        
        print(f"   ⏱️ Durée: {duration_ms:.0f}ms")
        print(f"   📊 Segments détectés: {len(segments_list)}")
        print(f"   ⚡ RTF: {(duration_ms/1000)/duration:.3f}")
        
        # Succès si détecte plus de segments
        if len(segments_list) > 3:
            print(f"   ✅ CORRECTION VAD FONCTIONNE!")
            print(f"   🎯 Détection améliorée: {len(segments_list)} segments")
            return True
        else:
            print(f"   ⚠️ Toujours insuffisant: {len(segments_list)} segments")
            return False
        
    except Exception as e:
        print(f"   ❌ Erreur VAD corrigé: {e}")
        return False

def main():
    """Test diagnostic complet"""
    
    print("🔧 DIAGNOSTIC STT SIMPLE - SUPERWHISPER V6 PHASE 4")
    print("Mission: Valider la correction VAD avec faster-whisper direct")
    
    try:
        # Test 1: Audio court
        success1 = test_audio_court()
        
        if success1:
            # Test 2: Audio long
            success2 = test_audio_long()
            
            if success2:
                # Test 3: Validation correction VAD
                success3 = test_vad_validation()
                
                if success3:
                    print(f"\n🎊 CORRECTION VAD VALIDÉE AVEC SUCCÈS!")
                    print(f"   ✅ faster-whisper fonctionne correctement")
                    print(f"   ✅ Paramètres VAD corrigés opérationnels")
                    print(f"   ✅ Prêt pour tests avec microphone réel")
                else:
                    print(f"\n⚠️ PROBLÈME PERSISTE DANS PARAMÈTRES VAD")
            else:
                print(f"\n⚠️ PROBLÈME DÉTECTÉ AVEC AUDIO LONG")
        else:
            print(f"\n❌ PROBLÈME DÉTECTÉ AVEC AUDIO COURT")
            
    except Exception as e:
        print(f"\n❌ Erreur diagnostic: {e}")
    
    print(f"\n🏁 Diagnostic terminé")

if __name__ == "__main__":
    main() 