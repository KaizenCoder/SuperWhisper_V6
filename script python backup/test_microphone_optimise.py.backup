#!/usr/bin/env python3
"""
Test Microphone Optimisé - SuperWhisper V6 Phase 4
🎯 VALIDATION: Transcription VAD avec gestion robuste erreurs

Mission: Tester transcription complète avec timeout adapté pour texte long
"""

import os
import sys
import time
import asyncio
import json
import sounddevice as sd
import numpy as np
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 SuperWhisper V6 Phase 4 STT - Test Microphone Optimisé")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter STT au path
sys.path.append(str(Path(__file__).parent.parent))

from STT.backends.prism_stt_backend import PrismSTTBackend

def validate_rtx3090_stt():
    """Validation systématique RTX 3090 pour STT"""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise pour STT")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée pour STT: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

async def test_microphone_simple():
    """Test microphone avec backend direct pour éviter timeouts"""
    
    print("\n" + "="*60)
    print("🎤 TEST MICROPHONE SIMPLE - BACKEND DIRECT")
    print("="*60)
    
    # Validation GPU
    validate_rtx3090_stt()
    
    # Configuration Backend direct (plus robuste)
    config = {
        'model': 'large-v2',
        'compute_type': 'float16',
        'language': 'fr',
        'beam_size': 5,
        'vad_filter': True
    }
    
    print(f"🚀 Initialisation Backend Prism direct...")
    backend = PrismSTTBackend(config)
    print(f"✅ Backend initialisé")
    
    # Configuration audio
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    
    print(f"\n📋 INSTRUCTIONS:")
    print(f"   1. Parlez clairement pendant 10-15 secondes")
    print(f"   2. Testez une phrase complète et complexe")
    print(f"   3. CTRL+C pour arrêter quand terminé")
    
    input(f"\n🎤 Appuyez sur ENTRÉE pour démarrer l'enregistrement...")
    
    print(f"\n🔴 ENREGISTREMENT EN COURS")
    print(f"   ⏹️ CTRL+C pour arrêter")
    
    # Enregistrement avec arrêt manuel
    frames = []
    try:
        # Configuration stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SIZE
        )
        
        with stream:
            start_recording = time.time()
            while True:
                # Lire chunk audio
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                frames.append(audio_chunk.flatten())
                
                # Affichage temps écoulé
                temps_ecoule = time.time() - start_recording
                print(f"\r⏱️ Temps écoulé: {temps_ecoule:.1f}s", end="", flush=True)
                
                # Arrêt automatique après 60s
                if temps_ecoule > 60:
                    print(f"\n⏰ Arrêt automatique après 60s")
                    break
                    
    except KeyboardInterrupt:
        duree_enregistrement = time.time() - start_recording
        print(f"\n✅ Enregistrement arrêté après {duree_enregistrement:.1f}s")
    
    # Construire audio final
    if not frames:
        print(f"❌ Aucun audio enregistré")
        return
    
    audio_complet = np.concatenate(frames)
    duree_audio = len(audio_complet) / SAMPLE_RATE
    niveau_audio = np.max(np.abs(audio_complet))
    
    print(f"\n📊 AUDIO ENREGISTRÉ:")
    print(f"   ⏱️ Durée: {duree_audio:.1f}s")
    print(f"   📊 Niveau: {niveau_audio:.3f}")
    print(f"   📏 Échantillons: {len(audio_complet)}")
    
    if niveau_audio < 0.01:
        print(f"⚠️ Niveau audio très faible, vérifiez votre microphone")
        return
    
    # Transcription STT DIRECTE avec timeout adapté
    print(f"\n🤖 TRANSCRIPTION STT AVEC VAD CORRIGÉ...")
    print(f"   🔧 Backend direct pour plus de robustesse")
    print(f"   ⏱️ Timeout adapté: {duree_audio * 5:.1f}s")
    
    start_transcription = time.time()
    
    try:
        # Utilisation directe du backend pour éviter timeouts UnifiedSTTManager
        result = await backend.transcribe(audio_complet)
        
        duree_transcription = time.time() - start_transcription
        rtf = duree_transcription / duree_audio
        
        print(f"✅ Transcription terminée en {duree_transcription:.2f}s")
        
        # Analyser résultats
        texte_transcrit = result.text
        nb_mots_transcrit = len(texte_transcrit.split()) if texte_transcrit else 0
        
        print(f"\n" + "="*60)
        print(f"📊 RÉSULTATS TRANSCRIPTION VAD CORRIGÉ")
        print(f"="*60)
        
        print(f"\n📝 TRANSCRIPTION OBTENUE:")
        print(f"   '{texte_transcrit}'")
        
        print(f"\n📊 MÉTRIQUES:")
        print(f"   ⏱️ Durée transcription: {duree_transcription:.2f}s")
        print(f"   ⚡ RTF: {rtf:.3f}")
        print(f"   🎯 Succès: {result.success}")
        print(f"   💪 Confiance: {result.confidence:.3f}")
        print(f"   🎮 Backend: {result.backend_used}")
        print(f"   📏 Mots transcrits: {nb_mots_transcrit}")
        
        print(f"\n🔧 ANALYSE VAD:")
        print(f"   📊 Segments détectés: {len(result.segments)}")
        
        if result.segments:
            print(f"   🔍 Détail segments (premiers 3):")
            for i, segment in enumerate(result.segments[:3]):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                duree_seg = end - start
                texte_seg = segment.get('text', '')
                print(f"      Seg {i+1}: {start:.1f}s → {end:.1f}s ({duree_seg:.1f}s) - '{texte_seg[:50]}...'")
        
        # Évaluation correction VAD
        print(f"\n🔧 ÉVALUATION CORRECTION VAD:")
        
        correction_ok = True
        
        if not texte_transcrit.strip():
            print(f"❌ Transcription vide")
            correction_ok = False
        else:
            print(f"✅ Transcription non vide")
        
        if nb_mots_transcrit < 5:
            print(f"❌ Trop peu de mots: {nb_mots_transcrit}")
            correction_ok = False
        else:
            print(f"✅ Nombre de mots significatif: {nb_mots_transcrit}")
        
        if len(result.segments) < 2:
            print(f"⚠️ Peu de segments VAD: {len(result.segments)}")
        else:
            print(f"✅ Segments VAD multiples: {len(result.segments)}")
        
        if rtf > 1.5:
            print(f"⚠️ RTF élevé: {rtf:.3f}")
        else:
            print(f"✅ RTF acceptable: {rtf:.3f}")
        
        # Résultat final
        print(f"\n🎯 RÉSULTAT:")
        if correction_ok:
            print(f"✅ CORRECTION VAD FONCTIONNE")
            print(f"   Transcription complète obtenue")
        else:
            print(f"⚠️ PROBLÈME DÉTECTÉ")
            print(f"   Transcription incomplète ou vide")
        
        # Sauvegarde
        resultats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duree_audio": duree_audio,
            "niveau_audio": niveau_audio,
            "texte_transcrit": texte_transcrit,
            "nb_mots_transcrit": nb_mots_transcrit,
            "duree_transcription": duree_transcription,
            "rtf": rtf,
            "confiance": result.confidence,
            "backend_used": result.backend_used,
            "nb_segments": len(result.segments),
            "correction_vad_ok": correction_ok
        }
        
        os.makedirs('test_output', exist_ok=True)
        output_file = f"test_output/test_microphone_optimise_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resultats, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")
        
        return correction_ok
        
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Point d'entrée principal"""
    print("🎤 TEST MICROPHONE OPTIMISÉ - SUPERWHISPER V6 PHASE 4")
    print("Mission: Validation robuste correction VAD")
    
    try:
        success = await test_microphone_simple()
        
        if success:
            print(f"\n🎊 TEST RÉUSSI - CORRECTION VAD VALIDÉE !")
        else:
            print(f"\n⚠️ TEST PARTIEL - À INVESTIGUER")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur test: {e}")
    
    print(f"\n✅ Test microphone optimisé terminé")

if __name__ == "__main__":
    asyncio.run(main()) 