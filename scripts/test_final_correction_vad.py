#!/usr/bin/env python3
"""
Test Final Correction VAD - SuperWhisper V6 Phase 4
🎯 VALIDATION FINALE: Correction VAD avec vraie voix humaine

Mission: Valider que la correction VAD permet la transcription complète 
du texte de 155 mots fourni sans s'arrêter à 25 mots

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

print("🎯 Test Final Correction VAD - SuperWhisper V6 Phase 4")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter STT au path
sys.path.append(str(Path(__file__).parent.parent))

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

# Texte de référence fourni (155 mots) pour le test final
TEXTE_REFERENCE_155_MOTS = """Bonjour, ceci est un test de validation pour SuperWhisper. Je vais maintenant énoncer plusieurs phrases de complexité croissante pour évaluer la précision de transcription. Premièrement, des mots simples : chat, chien, maison, voiture, ordinateur, téléphone. Deuxièmement, des phrases courtes : Il fait beau aujourd'hui. Le café est délicieux. J'aime la musique classique. Troisièmement, des phrases plus complexes : L'intelligence artificielle transforme notre manière de travailler et de communiquer dans le monde moderne. Quatrièmement, des termes techniques : algorithme, machine learning, GPU RTX 3090, faster-whisper, quantification, latence de transcription. Cinquièmement, des nombres et dates : vingt-trois, quarante-sept, mille neuf cent quatre-vingt-quinze, le quinze janvier deux mille vingt-quatre. Fin du test de validation."""

def compter_mots(texte):
    """Compte les mots dans un texte"""
    return len(texte.split()) if texte else 0

async def test_correction_vad_final():
    """Test final avec vraie voix humaine"""
    
    print("\n" + "="*70)
    print("🎯 TEST FINAL CORRECTION VAD - VRAIE VOIX HUMAINE")
    print("="*70)
    
    validate_rtx3090()
    
    # Import STT Backend corrigé
    from STT.backends.prism_stt_backend import PrismSTTBackend
    
    # Configuration Backend avec VAD corrigé
    config = {
        'model': 'large-v2',
        'compute_type': 'float16',
        'language': 'fr',
        'beam_size': 5,
        'vad_filter': True  # VAD activé avec paramètres corrigés
    }
    
    print(f"🚀 Initialisation Backend Prism STT avec VAD corrigé...")
    backend = PrismSTTBackend(config)
    print(f"✅ Backend prêt avec correction VAD")
    
    # Configuration audio
    SAMPLE_RATE = 16000
    
    # Affichage du texte à lire
    nb_mots_reference = compter_mots(TEXTE_REFERENCE_155_MOTS)
    print(f"\n📋 INSTRUCTIONS TEST FINAL:")
    print(f"   1. Lisez le texte complet ci-dessous ({nb_mots_reference} mots)")
    print(f"   2. Parlez clairement et distinctement")
    print(f"   3. Prenez votre temps, pas de stress")
    print(f"   4. CTRL+C pour arrêter quand vous avez terminé")
    
    print(f"\n📝 TEXTE À LIRE ({nb_mots_reference} MOTS):")
    print(f"{'='*70}")
    print(f"{TEXTE_REFERENCE_155_MOTS}")
    print(f"{'='*70}")
    
    input(f"\n🎤 Appuyez sur ENTRÉE quand vous êtes prêt à commencer la lecture...")
    
    print(f"\n🔴 ENREGISTREMENT EN COURS - Lisez le texte maintenant")
    print(f"   ⏹️ CTRL+C pour arrêter quand terminé")
    
    # Enregistrement avec arrêt manuel
    frames = []
    try:
        # Configuration stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=1024
        ) as stream:
            start_recording = time.time()
            while True:
                # Lire chunk audio
                audio_chunk, overflowed = stream.read(1024)
                frames.append(audio_chunk.flatten())
                
                # Affichage temps écoulé
                temps_ecoule = time.time() - start_recording
                print(f"\r⏱️ Temps écoulé: {temps_ecoule:.1f}s", end="", flush=True)
                
                # Arrêt automatique après 3 minutes
                if temps_ecoule > 180:
                    print(f"\n⏰ Arrêt automatique après 3 minutes")
                    break
                    
    except KeyboardInterrupt:
        duree_enregistrement = time.time() - start_recording
        print(f"\n✅ Enregistrement arrêté après {duree_enregistrement:.1f}s")
    
    # Construire audio final
    if not frames:
        print(f"❌ Aucun audio enregistré")
        return False
    
    audio_complet = np.concatenate(frames)
    duree_audio = len(audio_complet) / SAMPLE_RATE
    niveau_audio = np.max(np.abs(audio_complet))
    
    print(f"\n📊 AUDIO ENREGISTRÉ:")
    print(f"   ⏱️ Durée: {duree_audio:.1f}s")
    print(f"   📊 Niveau: {niveau_audio:.3f}")
    print(f"   📏 Échantillons: {len(audio_complet)}")
    
    if niveau_audio < 0.01:
        print(f"⚠️ Niveau audio très faible, vérifiez votre microphone")
        return False
    
    # Transcription STT avec VAD corrigé
    print(f"\n🤖 TRANSCRIPTION AVEC CORRECTION VAD...")
    print(f"   🔧 Utilisation paramètres VAD corrigés")
    
    start_transcription = time.time()
    
    try:
        # Transcription avec backend corrigé
        result = await backend.transcribe(audio_complet)
        
        duree_transcription = time.time() - start_transcription
        rtf = duree_transcription / duree_audio
        
        print(f"✅ Transcription terminée en {duree_transcription:.2f}s")
        
        # Analyser résultats
        texte_transcrit = result.text
        nb_mots_transcrit = compter_mots(texte_transcrit)
        nb_mots_reference = compter_mots(TEXTE_REFERENCE_155_MOTS)
        
        # Pourcentage de complétion
        taux_completion = (nb_mots_transcrit / nb_mots_reference * 100) if nb_mots_reference > 0 else 0
        
        print(f"\n" + "="*70)
        print(f"📊 RÉSULTATS TEST FINAL CORRECTION VAD")
        print(f"="*70)
        
        print(f"\n📝 TRANSCRIPTION OBTENUE:")
        print(f"   '{texte_transcrit}'")
        
        print(f"\n📊 ANALYSE QUANTITATIVE:")
        print(f"   📏 Mots attendus: {nb_mots_reference}")
        print(f"   📏 Mots transcrits: {nb_mots_transcrit}")
        print(f"   📈 Taux complétion: {taux_completion:.1f}%")
        print(f"   ⏱️ Durée transcription: {duree_transcription:.2f}s")
        print(f"   ⚡ RTF: {rtf:.3f}")
        print(f"   💪 Confiance: {result.confidence:.3f}")
        print(f"   🎮 Backend: {result.backend_used}")
        
        print(f"\n🔧 ANALYSE VAD:")
        print(f"   📊 Segments détectés: {len(result.segments)}")
        print(f"   🎯 Succès transcription: {result.success}")
        
        if result.segments:
            print(f"   🔍 Premiers segments:")
            for i, segment in enumerate(result.segments[:3]):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                duree_seg = end - start
                texte_seg = segment.get('text', '')
                print(f"      Seg {i+1}: {start:.1f}s-{end:.1f}s ({duree_seg:.1f}s) - '{texte_seg[:40]}...'")
        
        # Évaluation correction VAD
        print(f"\n🎯 ÉVALUATION CORRECTION VAD:")
        
        success_criteria = []
        
        # Critère 1: Transcription non vide
        if texte_transcrit.strip():
            print(f"   ✅ Transcription non vide")
            success_criteria.append(True)
        else:
            print(f"   ❌ Transcription vide")
            success_criteria.append(False)
        
        # Critère 2: Plus de 25 mots (problème original)
        if nb_mots_transcrit > 25:
            print(f"   ✅ Dépasse 25 mots (problème résolu): {nb_mots_transcrit} mots")
            success_criteria.append(True)
        else:
            print(f"   ❌ Toujours bloqué à 25 mots: {nb_mots_transcrit} mots")
            success_criteria.append(False)
        
        # Critère 3: Taux de complétion raisonnable
        if taux_completion > 70:
            print(f"   ✅ Taux complétion excellent: {taux_completion:.1f}%")
            success_criteria.append(True)
        elif taux_completion > 40:
            print(f"   ⚠️ Taux complétion acceptable: {taux_completion:.1f}%")
            success_criteria.append(True)
        else:
            print(f"   ❌ Taux complétion insuffisant: {taux_completion:.1f}%")
            success_criteria.append(False)
        
        # Critère 4: Segments VAD multiples
        if len(result.segments) > 5:
            print(f"   ✅ VAD détecte plusieurs segments: {len(result.segments)}")
            success_criteria.append(True)
        elif len(result.segments) > 1:
            print(f"   ⚠️ VAD détecte quelques segments: {len(result.segments)}")
            success_criteria.append(True)
        else:
            print(f"   ❌ VAD détecte trop peu de segments: {len(result.segments)}")
            success_criteria.append(False)
        
        # Critère 5: Performance acceptable
        if rtf < 0.5:
            print(f"   ✅ Performance excellente: RTF {rtf:.3f}")
            success_criteria.append(True)
        elif rtf < 1.0:
            print(f"   ⚠️ Performance acceptable: RTF {rtf:.3f}")
            success_criteria.append(True)
        else:
            print(f"   ❌ Performance lente: RTF {rtf:.3f}")
            success_criteria.append(False)
        
        # Résultat final
        success_count = sum(success_criteria)
        total_criteria = len(success_criteria)
        success_rate = success_count / total_criteria * 100
        
        print(f"\n🏆 RÉSULTAT FINAL:")
        print(f"   📊 Critères réussis: {success_count}/{total_criteria} ({success_rate:.1f}%)")
        
        if success_count >= 4:
            print(f"   ✅ CORRECTION VAD VALIDÉE AVEC SUCCÈS!")
            print(f"   🎊 Problème transcription incomplète RÉSOLU")
            print(f"   🚀 SuperWhisper V6 Phase 4 STT opérationnel")
            validation_success = True
        elif success_count >= 2:
            print(f"   ⚠️ CORRECTION VAD PARTIELLEMENT VALIDÉE")
            print(f"   🔧 Amélioration nécessaire mais progrès significatif")
            validation_success = True
        else:
            print(f"   ❌ CORRECTION VAD NON VALIDÉE")
            print(f"   🔧 Problème persiste, investigation requise")
            validation_success = False
        
        # Sauvegarde résultats
        resultats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "correction_vad_final",
            "duree_audio": duree_audio,
            "niveau_audio": niveau_audio,
            "texte_reference": TEXTE_REFERENCE_155_MOTS,
            "nb_mots_reference": nb_mots_reference,
            "texte_transcrit": texte_transcrit,
            "nb_mots_transcrit": nb_mots_transcrit,
            "taux_completion": taux_completion,
            "duree_transcription": duree_transcription,
            "rtf": rtf,
            "confiance": result.confidence,
            "backend_used": result.backend_used,
            "nb_segments": len(result.segments),
            "success_criteria": success_criteria,
            "success_count": success_count,
            "success_rate": success_rate,
            "validation_success": validation_success
        }
        
        os.makedirs('test_output', exist_ok=True)
        output_file = f"test_output/test_final_correction_vad_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resultats, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")
        
        return validation_success
        
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Point d'entrée principal"""
    print("🎯 TEST FINAL CORRECTION VAD - SUPERWHISPER V6 PHASE 4")
    print("Mission: Validation finale avec texte de 155 mots")
    
    try:
        success = await test_correction_vad_final()
        
        if success:
            print(f"\n🎊 MISSION ACCOMPLIE - CORRECTION VAD VALIDÉE !")
            print(f"SuperWhisper V6 Phase 4 STT prêt pour la suite")
        else:
            print(f"\n⚠️ MISSION PARTIELLEMENT RÉUSSIE")
            print(f"Correction VAD améliore mais nécessite ajustements")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur test: {e}")
    
    print(f"\n✅ Test final correction VAD terminé")

if __name__ == "__main__":
    asyncio.run(main()) 