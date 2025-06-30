#!/usr/bin/env python3
"""
Test Validation Texte Fourni Complet - SuperWhisper V6 Phase 4
🔧 VALIDATION: Texte complet 155 mots pour vérifier correction VAD

Mission: Valider que la correction VAD permet de transcrire
le texte fourni COMPLET (155 mots) au lieu de s'arrêter à 25 mots.
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

print("🎮 Test Validation Texte Fourni - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter STT au path
sys.path.append(str(Path(__file__).parent.parent))

from STT.unified_stt_manager import UnifiedSTTManager

# =============================================================================
# 📝 TEXTE FOURNI COMPLET - 155 MOTS À TRANSCRIRE INTÉGRALEMENT
# =============================================================================
TEXTE_FOURNI_155_MOTS = """
Dans le cadre du développement de SuperWhisper V6, nous procédons à l'intégration du module Speech-to-Text 
utilisant Prism_Whisper2 optimisé pour la configuration RTX 3090. Cette phase critique nécessite une 
validation rigoureuse des paramètres Voice Activity Detection pour assurer une transcription complète 
et précise. Le système doit être capable de traiter des phrases complexes, techniques et longues sans 
interruption prématurée. L'architecture implémentée comprend un gestionnaire unifié avec fallback 
automatique, un cache LRU optimisé et des circuit breakers pour la robustesse. Les performances ciblées 
incluent une latence inférieure à quatre cents millisecondes pour cinq secondes d'audio et un facteur 
temps réel inférieur à un. La configuration GPU utilise exclusivement la RTX 3090 via CUDA_VISIBLE_DEVICES 
pour garantir l'allocation mémoire optimale. Ce test de validation doit confirmer que tous ces éléments 
fonctionnent harmonieusement ensemble pour produire une transcription fidèle et complète du texte prononcé.
"""

def compter_mots(texte: str) -> int:
    """Compte le nombre de mots dans un texte"""
    return len(texte.split())

def calculer_precision_mots(texte_original: str, texte_transcrit: str) -> float:
    """Calcule la précision au niveau des mots"""
    if not texte_transcrit.strip():
        return 0.0
    
    mots_originaux = set(texte_original.lower().split())
    mots_transcrits = set(texte_transcrit.lower().split())
    
    intersection = mots_originaux.intersection(mots_transcrits)
    return len(intersection) / len(mots_originaux) * 100

def analyser_segments_transcrits(result_segments: list) -> dict:
    """Analyse les segments transcrits pour diagnostiquer VAD"""
    if not result_segments:
        return {
            "nombre_segments": 0,
            "duree_totale": 0.0,
            "segments_details": []
        }
    
    segments_details = []
    duree_totale = 0.0
    
    for segment in result_segments:
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        duree = end - start
        duree_totale += duree
        
        segments_details.append({
            "start": start,
            "end": end,
            "duree": duree,
            "texte": segment.get('text', ''),
            "longueur_texte": len(segment.get('text', ''))
        })
    
    return {
        "nombre_segments": len(result_segments),
        "duree_totale": duree_totale,
        "duree_moyenne_segment": duree_totale / len(result_segments),
        "segments_details": segments_details
    }

async def test_validation_texte_fourni():
    """
    🎯 TEST VALIDATION TEXTE FOURNI COMPLET
    Teste la correction VAD avec le texte de 155 mots
    """
    
    print("\n" + "="*80)
    print("🔧 TEST VALIDATION TEXTE FOURNI COMPLET - 155 MOTS")
    print("="*80)
    
    # Analyser texte fourni
    nb_mots_original = compter_mots(TEXTE_FOURNI_155_MOTS)
    print(f"\n📋 TEXTE FOURNI À TRANSCRIRE:")
    print(f"   📏 Longueur: {len(TEXTE_FOURNI_155_MOTS)} caractères")
    print(f"   🔢 Nombre de mots: {nb_mots_original}")
    print(f"   ⏱️ Durée estimée lecture: {nb_mots_original * 0.6:.1f}s (150 mots/min)")
    
    print(f"\n📝 CONTENU:")
    print(f"   {TEXTE_FOURNI_155_MOTS.strip()}")
    
    print(f"\n🎯 OBJECTIF VALIDATION:")
    print(f"   Transcrire les {nb_mots_original} mots COMPLETS")
    print(f"   (Problème résolu: coupure après 25 mots)")
    
    # Initialisation STT
    print(f"\n🚀 Initialisation UnifiedSTTManager...")
    stt_manager = UnifiedSTTManager()
    print(f"✅ STT Manager prêt")
    
    # Configuration audio
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    
    print(f"\n📋 INSTRUCTIONS LECTURE:")
    print(f"   1. Lisez le texte COMPLET à voix haute")
    print(f"   2. Articulez clairement mais naturellement") 
    print(f"   3. Prenez votre temps, PAS de précipitation")
    print(f"   4. Le test s'arrête automatiquement après 120s")
    print(f"   5. Ou appuyez sur CTRL+C pour arrêter manuellement")
    
    input(f"\n🎤 Appuyez sur ENTRÉE quand vous êtes prêt à lire le texte complet...")
    
    print(f"\n🔴 ENREGISTREMENT EN COURS - LISEZ LE TEXTE COMPLET")
    print(f"   📋 Texte à lire: {TEXTE_FOURNI_155_MOTS.strip()}")
    print(f"   ⏹️ CTRL+C pour arrêter quand vous avez terminé")
    
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
                
                # Arrêt automatique après 120s
                if temps_ecoule > 120:
                    print(f"\n⏰ Arrêt automatique après 120s")
                    break
                    
    except KeyboardInterrupt:
        duree_enregistrement = time.time() - start_recording
        print(f"\n✅ Enregistrement arrêté manuellement après {duree_enregistrement:.1f}s")
    
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
    
    # Transcription STT avec VAD corrigé
    print(f"\n🤖 TRANSCRIPTION STT AVEC VAD CORRIGÉ...")
    print(f"   🔧 Paramètres VAD optimisés pour texte long")
    
    start_transcription = time.time()
    
    try:
        result = await stt_manager.transcribe(audio_complet)
        
        duree_transcription = time.time() - start_transcription
        rtf = duree_transcription / duree_audio
        
        print(f"✅ Transcription terminée en {duree_transcription:.2f}s")
        
        # Analyser résultats
        texte_transcrit = result.text
        nb_mots_transcrit = compter_mots(texte_transcrit)
        precision_mots = calculer_precision_mots(TEXTE_FOURNI_155_MOTS, texte_transcrit)
        
        # Analyser segments VAD
        segments_analyse = analyser_segments_transcrits(result.segments)
        
        print(f"\n" + "="*80)
        print(f"📊 RÉSULTATS VALIDATION TEXTE FOURNI COMPLET")
        print(f"="*80)
        
        print(f"\n📝 TRANSCRIPTION OBTENUE:")
        print(f"   '{texte_transcrit}'")
        
        print(f"\n📊 MÉTRIQUES TRANSCRIPTION:")
        print(f"   ⏱️ Durée transcription: {duree_transcription:.2f}s")
        print(f"   ⚡ RTF: {rtf:.3f}")
        print(f"   🎯 Succès: {result.success}")
        print(f"   💪 Confiance: {result.confidence:.3f}")
        print(f"   🎮 Backend: {result.backend_used}")
        
        print(f"\n📏 ANALYSE LONGUEUR:")
        print(f"   📋 Mots originaux: {nb_mots_original}")
        print(f"   📝 Mots transcrits: {nb_mots_transcrit}")
        print(f"   📈 Pourcentage: {(nb_mots_transcrit/nb_mots_original)*100:.1f}%")
        print(f"   🎯 Précision mots: {precision_mots:.1f}%")
        
        print(f"\n🔧 ANALYSE VAD (Voice Activity Detection):")
        print(f"   📊 Segments détectés: {segments_analyse['nombre_segments']}")
        print(f"   ⏱️ Durée totale segments: {segments_analyse['duree_totale']:.1f}s")
        if segments_analyse['nombre_segments'] > 0:
            print(f"   📈 Durée moyenne/segment: {segments_analyse['duree_moyenne_segment']:.1f}s")
        
        # Diagnostic VAD détaillé
        if segments_analyse['nombre_segments'] > 0:
            print(f"\n🔍 DÉTAIL SEGMENTS VAD:")
            for i, segment in enumerate(segments_analyse['segments_details'][:5]):  # Max 5 premiers
                print(f"   Segment {i+1}: {segment['start']:.1f}s → {segment['end']:.1f}s "
                      f"({segment['duree']:.1f}s) - {len(segment['texte'])} chars")
            
            if len(segments_analyse['segments_details']) > 5:
                print(f"   ... et {len(segments_analyse['segments_details'])-5} autres segments")
        
        # Évaluation correction VAD
        print(f"\n" + "="*80)
        print(f"🔧 ÉVALUATION CORRECTION VAD")
        print(f"="*80)
        
        # Critères de réussite
        correction_reussie = True
        problemes = []
        
        # 1. Transcription non vide
        if not texte_transcrit.strip():
            correction_reussie = False
            problemes.append("Transcription vide")
        else:
            print(f"✅ Transcription non vide")
        
        # 2. Percentage de mots significatif (> 50%)
        pourcentage_mots = (nb_mots_transcrit/nb_mots_original)*100
        if pourcentage_mots < 50:
            correction_reussie = False
            problemes.append(f"Trop peu de mots transcrits ({pourcentage_mots:.1f}%)")
        else:
            print(f"✅ Nombre de mots significatif: {pourcentage_mots:.1f}%")
        
        # 3. Plusieurs segments détectés
        if segments_analyse['nombre_segments'] < 3:
            correction_reussie = False
            problemes.append(f"Trop peu de segments VAD ({segments_analyse['nombre_segments']})")
        else:
            print(f"✅ Segments VAD multiples: {segments_analyse['nombre_segments']}")
        
        # 4. RTF raisonnable
        if rtf > 2.0:
            correction_reussie = False
            problemes.append(f"RTF trop élevé ({rtf:.2f})")
        else:
            print(f"✅ RTF acceptable: {rtf:.3f}")
        
        # 5. Précision mots raisonnable
        if precision_mots < 30:
            correction_reussie = False
            problemes.append(f"Précision trop faible ({precision_mots:.1f}%)")
        else:
            print(f"✅ Précision mots acceptable: {precision_mots:.1f}%")
        
        # Résultat final
        print(f"\n🎯 RÉSULTAT VALIDATION CORRECTION VAD:")
        if correction_reussie:
            print(f"✅ CORRECTION VAD VALIDÉE AVEC SUCCÈS !")
            print(f"   La transcription complète fonctionne correctement")
            statut = "VALIDÉ"
        else:
            print(f"❌ CORRECTION VAD À AMÉLIORER")
            print(f"   Problèmes détectés:")
            for probleme in problemes:
                print(f"   - {probleme}")
            statut = "À_CORRIGER"
        
        # Sauvegarde résultats
        resultats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "texte_original": TEXTE_FOURNI_155_MOTS.strip(),
            "nb_mots_original": nb_mots_original,
            "texte_transcrit": texte_transcrit,
            "nb_mots_transcrit": nb_mots_transcrit,
            "pourcentage_mots": pourcentage_mots,
            "precision_mots": precision_mots,
            "duree_audio": duree_audio,
            "duree_transcription": duree_transcription,
            "rtf": rtf,
            "confiance": result.confidence,
            "backend_used": result.backend_used,
            "segments_analyse": segments_analyse,
            "correction_vad_validee": correction_reussie,
            "problemes": problemes,
            "statut": statut
        }
        
        # Créer dossier de sortie
        os.makedirs('test_output', exist_ok=True)
        
        # Sauvegarder
        output_file = f"test_output/validation_texte_fourni_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resultats, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")
        
        return correction_reussie
        
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        return False

async def main():
    """Point d'entrée principal"""
    print("🔧 TEST VALIDATION TEXTE FOURNI COMPLET - SUPERWHISPER V6 PHASE 4")
    print("Mission: Valider correction VAD avec texte complet 155 mots")
    
    try:
        success = await test_validation_texte_fourni()
        
        if success:
            print(f"\n🎊 VALIDATION RÉUSSIE - CORRECTION VAD OPÉRATIONNELLE !")
        else:
            print(f"\n⚠️ VALIDATION PARTIELLE - AJUSTEMENTS RECOMMANDÉS")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur test: {e}")
    
    print(f"\n✅ Test de validation texte fourni terminé")

if __name__ == "__main__":
    asyncio.run(main()) 