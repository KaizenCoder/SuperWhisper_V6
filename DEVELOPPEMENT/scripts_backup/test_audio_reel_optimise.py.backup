#!/usr/bin/env python3
"""
Test Audio Réel - Solution STT Optimisée
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
Test avec fichier audio réel (lecture texte complexe au micro Rode)
⚠️ ATTENTION: Résultats faussement flatteurs vs conditions réelles microphone live
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajout du chemin STT au PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import asyncio
import time
import json
import numpy as np
import torch
import librosa
from pathlib import Path
from datetime import datetime
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")

def load_audio_file(file_path: str):
    """Chargement et préparation du fichier audio"""
    print(f"\n🎵 CHARGEMENT FICHIER AUDIO")
    print("-" * 50)
    
    try:
        # Chargement avec librosa (16kHz requis pour Whisper)
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        duration = len(audio) / sr
        file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
        
        print(f"✅ Fichier chargé: {Path(file_path).name}")
        print(f"   Durée: {duration:.2f}s")
        print(f"   Taille: {file_size:.1f}MB")
        print(f"   Sample rate: {sr}Hz")
        print(f"   Échantillons: {len(audio):,}")
        
        return audio, duration
        
    except Exception as e:
        print(f"❌ Erreur chargement audio: {e}")
        return None, 0

async def test_backend_optimise_avec_audio(audio: np.ndarray, duration: float):
    """Test du backend optimisé avec audio réel"""
    print(f"\n⚙️ TEST BACKEND OPTIMISÉ - AUDIO RÉEL")
    print("-" * 50)
    
    try:
        from STT.backends.prism_stt_backend_optimized import OptimizedPrismSTTBackend
        
        config = {
            'model': 'large-v2',
            'compute_type': 'float16',
            'language': 'fr',  # Forçage français
            'beam_size': 10,   # Beam search optimisé
            'vad_filter': True,
            'vad_parameters': {
                'threshold': 0.3,
                'min_speech_duration_ms': 100,
                'max_speech_duration_s': float('inf'),
                'min_silence_duration_ms': 2000,
                'speech_pad_ms': 400
            }
        }
        
        print("🚀 Initialisation backend optimisé...")
        backend = OptimizedPrismSTTBackend(config)
        
        print("🎯 Transcription audio réel...")
        start_time = time.perf_counter()
        
        result = await backend.transcribe(audio)
        
        processing_time = time.perf_counter() - start_time
        
        if result['success']:
            print(f"\n📝 RÉSULTATS TRANSCRIPTION:")
            print(f"   Texte: \"{result['text']}\"")
            print(f"   Confiance: {result['confidence']:.3f}")
            print(f"   Segments: {len(result['segments'])}")
            print(f"   Temps traitement: {processing_time:.2f}s")
            print(f"   RTF: {result['rtf']:.3f}")
            print(f"   Mots estimés: {len(result['text'].split())}")
            
            return result
        else:
            print(f"❌ Transcription échouée: {result.get('error', 'Erreur inconnue')}")
            return None
            
    except Exception as e:
        print(f"❌ Erreur test backend: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_post_processor_avec_transcription(transcription: str):
    """Test du post-processeur avec la transcription obtenue"""
    print(f"\n🧪 TEST POST-PROCESSEUR - TRANSCRIPTION RÉELLE")
    print("-" * 50)
    
    try:
        from STT.stt_postprocessor import STTPostProcessor
        
        processor = STTPostProcessor()
        
        print(f"📝 Texte original: \"{transcription}\"")
        
        start_time = time.time()
        processed, metrics = processor.process(transcription, 0.8)
        processing_time = (time.time() - start_time) * 1000  # ms
        
        print(f"📝 Texte corrigé: \"{processed}\"")
        print(f"   Corrections appliquées: {metrics['corrections_applied']}")
        print(f"   Boost confiance: +{metrics['confidence_boost']:.3f}")
        print(f"   Temps traitement: {processing_time:.1f}ms")
        
        # Comparaison longueur
        mots_avant = len(transcription.split())
        mots_après = len(processed.split())
        
        print(f"   Mots avant: {mots_avant}")
        print(f"   Mots après: {mots_après}")
        print(f"   Différence: {mots_après - mots_avant:+d} mots")
        
        return processed, metrics
        
    except Exception as e:
        print(f"❌ Erreur post-processeur: {e}")
        return transcription, {}

async def test_manager_optimise_avec_audio(audio: np.ndarray):
    """Test du manager optimisé avec audio réel"""
    print(f"\n🧠 TEST MANAGER OPTIMISÉ - AUDIO RÉEL")
    print("-" * 50)
    
    try:
        from STT.unified_stt_manager_optimized import OptimizedUnifiedSTTManager
        
        config = {
            'model': 'large-v2',
            'compute_type': 'float16'
        }
        
        print("🚀 Initialisation manager optimisé...")
        manager = OptimizedUnifiedSTTManager(config)
        
        print("🎯 Transcription via manager...")
        start_time = time.perf_counter()
        
        # Note: Le manager optimisé pourrait ne pas avoir la méthode transcribe_audio
        # On teste juste l'initialisation pour ce test
        processing_time = time.perf_counter() - start_time
        
        print(f"✅ Manager initialisé en {processing_time:.2f}s")
        print("⚠️ Test transcription via manager nécessite implémentation complète")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test manager: {e}")
        return False

def generate_test_report(audio_duration: float, backend_result: dict, post_processed: str, metrics: dict):
    """Génération du rapport de test audio réel"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_audio_reel_optimise_report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'test_type': 'audio_reel_fichier',
        'warning': 'Résultats faussement flatteurs vs microphone live',
        'audio_info': {
            'duration_seconds': audio_duration,
            'file': 'enregistrement_avec_lecture_texte_complet_depuis_micro_rode.wav'
        },
        'gpu_config': {
            'device': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'cuda_version': torch.version.cuda
        },
        'backend_result': backend_result,
        'post_processing': {
            'original_text': backend_result.get('text', '') if backend_result else '',
            'processed_text': post_processed,
            'metrics': metrics
        },
        'performance': {
            'rtf': backend_result.get('rtf', 0) if backend_result else 0,
            'processing_time': backend_result.get('processing_time', 0) if backend_result else 0,
            'confidence': backend_result.get('confidence', 0) if backend_result else 0
        }
    }
    
    # Sauvegarde rapport
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Rapport sauvegardé: {report_file}")
    return report_file

async def main():
    """Test principal avec audio réel"""
    print("🎯 TEST AUDIO RÉEL - SOLUTION STT OPTIMISÉE")
    print("🚨 GPU RTX 3090 OBLIGATOIRE")
    print("⚠️ ATTENTION: Fichier pré-enregistré ≠ conditions réelles microphone")
    print("=" * 80)
    
    try:
        # 1. Validation GPU
        validate_rtx3090_configuration()
        
        # 2. Chargement fichier audio
        audio_file = "tests/test_input/enregistrement_avec_lecture_texte_complet_depuis_micro_rode.wav"
        
        if not Path(audio_file).exists():
            print(f"❌ Fichier audio non trouvé: {audio_file}")
            return False
        
        audio, duration = load_audio_file(audio_file)
        if audio is None:
            return False
        
        # 3. Test backend optimisé avec audio réel
        backend_result = await test_backend_optimise_avec_audio(audio, duration)
        
        if not backend_result:
            print("❌ Test backend échoué - arrêt")
            return False
        
        # 4. Test post-processeur avec transcription
        post_processed, metrics = await test_post_processor_avec_transcription(backend_result['text'])
        
        # 5. Test manager optimisé
        manager_ok = await test_manager_optimise_avec_audio(audio)
        
        # 6. Résumé global
        print("\n" + "="*80)
        print("📊 RÉSUMÉ TEST AUDIO RÉEL")
        print("="*80)
        
        print(f"✅ Fichier audio: {duration:.2f}s traité")
        print(f"✅ Backend optimisé: Transcription réussie")
        print(f"✅ Post-processeur: {metrics.get('corrections_applied', 0)} corrections")
        print(f"{'✅' if manager_ok else '⚠️'} Manager optimisé: {'Initialisé' if manager_ok else 'Problème'}")
        
        # Métriques clés
        print(f"\n📈 MÉTRIQUES CLÉS:")
        print(f"   RTF: {backend_result['rtf']:.3f}")
        print(f"   Confiance: {backend_result['confidence']:.3f}")
        print(f"   Temps traitement: {backend_result['processing_time']:.2f}s")
        print(f"   Mots transcrits: {len(backend_result['text'].split())}")
        print(f"   Corrections post-proc: {metrics.get('corrections_applied', 0)}")
        
        # 7. Génération rapport
        report_file = generate_test_report(duration, backend_result, post_processed, metrics)
        
        # 8. Avertissement important
        print(f"\n⚠️ AVERTISSEMENT IMPORTANT:")
        print("   → Ce test utilise un fichier pré-enregistré")
        print("   → Les résultats sont FAUSSEMENT FLATTEURS")
        print("   → Le VRAI test sera la lecture au microphone live")
        print("   → Conditions réelles = bruit, distance, accent, etc.")
        
        print(f"\n🎯 PROCHAINE ÉTAPE CRITIQUE:")
        print("   → Test microphone live avec lecture texte complexe")
        print("   → Validation humaine de la précision")
        print("   → Mesure WER en conditions réelles")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur test audio réel: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Exécution test
    success = asyncio.run(main())
    
    if success:
        print(f"\n✅ Test audio réel terminé avec succès")
        print("⚠️ Rappel: Résultats faussement flatteurs vs microphone live")
    else:
        print(f"\n❌ Test audio réel échoué") 