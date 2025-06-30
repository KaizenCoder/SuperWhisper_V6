#!/usr/bin/env python3
"""
Test de Validation Correction VAD - Solution Experte
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch
import time
import json
from pathlib import Path
import numpy as np
from datetime import datetime

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

def test_correction_vad():
    """Test rapide de la correction VAD experte"""
    
    print("\n" + "="*60)
    print("🔧 TEST CORRECTION VAD - SOLUTION EXPERTE")
    print("="*60)
    
    # Validation GPU
    validate_rtx3090_configuration()
    
    # Import backend corrigé
    sys.path.append('.')
    from STT.backends.prism_stt_backend import PrismSTTBackend
    
    # Configuration backend
    config = {
        'model': 'large-v2',
        'device': 'cuda',
        'compute_type': 'float16',
        'language': 'fr',
        'beam_size': 5,
        'vad_filter': True
    }
    
    print(f"🤖 Initialisation Prism STT Backend...")
    start_init = time.time()
    backend = PrismSTTBackend(config)
    init_time = time.time() - start_init
    print(f"✅ Backend initialisé en {init_time:.2f}s")
    
    # Charger audio de validation
    validation_file = Path("test_output/validation_texte_fourni.json")
    if not validation_file.exists():
        print("❌ Fichier validation non trouvé")
        return False
    
    with open(validation_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    texte_reference = validation_data['test_info']['texte_reference']
    mots_reference = validation_data['analyse_precision']['mots_reference']
    
    print(f"📝 Texte référence: {mots_reference} mots")
    print(f"🎯 Objectif: Transcription complète (155/155 mots)")
    
    # Charger audio depuis fichier WAV
    audio_file = Path("test_output/audio_validation.wav")
    if audio_file.exists():
        import soundfile as sf
        audio_data, sample_rate = sf.read(str(audio_file))
        print(f"🎵 Audio chargé: {len(audio_data)} samples, {sample_rate}Hz")
        
        # Convertir vers format backend
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Mono
        if sample_rate != 16000:
            import resampy
            audio_data = resampy.resample(audio_data, sample_rate, 16000)
        
        audio_data = audio_data.astype(np.float32)
        duree_audio = len(audio_data) / 16000
        
    else:
        print("⚠️ Fichier audio non trouvé, utilisation audio synthétique")
        # Audio synthétique pour test
        duree_audio = 17.0  # 17 secondes comme validation
        samples = int(duree_audio * 16000)
        audio_data = np.random.normal(0, 0.1, samples).astype(np.float32)
    
    print(f"⏱️ Durée audio: {duree_audio:.1f}s")
    
    # Test transcription avec VAD corrigé
    print(f"\n🔧 TRANSCRIPTION AVEC VAD CORRIGÉ...")
    start_time = time.time()
    
    try:
        import asyncio
        result = asyncio.run(backend.transcribe(audio_data))
        
        duration_ms = (time.time() - start_time) * 1000
        
        if result.success:
            texte_transcrit = result.text.strip()
            mots_transcrits = len(texte_transcrit.split()) if texte_transcrit else 0
            
            print(f"✅ TRANSCRIPTION RÉUSSIE en {duration_ms:.0f}ms")
            print(f"📝 Texte transcrit ({mots_transcrits} mots):")
            print(f"   '{texte_transcrit}'")
            print(f"📊 Analyse:")
            print(f"   - Mots référence: {mots_reference}")
            print(f"   - Mots transcrits: {mots_transcrits}")
            print(f"   - Progression: {mots_transcrits}/{mots_reference} ({(mots_transcrits/mots_reference)*100:.1f}%)")
            print(f"   - Confiance: {result.confidence:.3f}")
            print(f"   - RTF: {result.rtf:.3f}")
            print(f"   - Latence: {duration_ms:.0f}ms")
            
            # Vérifier amélioration
            if mots_transcrits > 25:
                print(f"🎉 AMÉLIORATION DÉTECTÉE!")
                print(f"   Avant correction: 25 mots (16%)")
                print(f"   Après correction: {mots_transcrits} mots ({(mots_transcrits/mots_reference)*100:.1f}%)")
                
                if mots_transcrits >= mots_reference * 0.9:  # 90%+
                    print(f"✅ SUCCÈS COMPLET: Transcription quasi-complète!")
                    return True
                elif mots_transcrits >= mots_reference * 0.7:  # 70%+
                    print(f"✅ SUCCÈS PARTIEL: Amélioration significative!")
                    return True
                else:
                    print(f"⚠️ AMÉLIORATION PARTIELLE: Progrès mais peut être optimisé")
                    return True
            else:
                print(f"❌ AUCUNE AMÉLIORATION: Toujours {mots_transcrits} mots")
                return False
                
        else:
            print(f"❌ ÉCHEC TRANSCRIPTION: {result.error}")
            return False
            
    except Exception as e:
        print(f"❌ ERREUR TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test principal"""
    print("🔧 VALIDATION CORRECTION VAD EXPERTE - SUPERWHISPER V6")
    print("="*60)
    
    try:
        success = test_correction_vad()
        
        print("\n" + "="*60)
        if success:
            print("🎉 TEST RÉUSSI: Correction VAD efficace!")
            print("📋 Actions suggérées:")
            print("   1. Lancer validation complète: python scripts/test_validation_texte_fourni.py")
            print("   2. Tester avec microphone: python scripts/test_microphone_optimise.py")
            print("   3. Procéder Phase 5 si résultats satisfaisants")
        else:
            print("❌ TEST ÉCHOUÉ: Correction VAD insuffisante")
            print("📋 Actions requises:")
            print("   1. Vérifier paramètres VAD dans prism_stt_backend.py")
            print("   2. Analyser logs d'erreur")
            print("   3. Contacter expert si blocage persiste")
        print("="*60)
        
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 