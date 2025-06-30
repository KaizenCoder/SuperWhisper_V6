#!/usr/bin/env python3
"""
Test STT avec microphone réel - VALIDATION HUMAINE OBLIGATOIRE
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 SuperWhisper V6 Phase 4 STT - Test Microphone Réel")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import asyncio
import time
import json
import numpy as np
from datetime import datetime
import sounddevice as sd
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from STT.unified_stt_manager import UnifiedSTTManager

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

def enregistrer_audio_microphone(duree_secondes=5, sample_rate=16000):
    """
    Enregistre audio depuis le microphone système
    
    Args:
        duree_secondes: Durée d'enregistrement
        sample_rate: Fréquence d'échantillonnage
        
    Returns:
        np.ndarray: Audio enregistré en float32
    """
    print(f"\n🎤 ENREGISTREMENT MICROPHONE - {duree_secondes}s")
    print("=" * 50)
    
    # Vérifier les périphériques audio disponibles
    print("🔍 Périphériques audio disponibles :")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"   {i}: {device['name']} (entrées: {device['max_input_channels']})")
    
    print(f"\n🎙️ Préparez-vous à parler pendant {duree_secondes} secondes...")
    input("   Appuyez sur ENTRÉE quand vous êtes prêt...")
    
    print(f"\n🔴 ENREGISTREMENT EN COURS... ({duree_secondes}s)")
    print("   Parlez maintenant dans votre microphone !")
    
    try:
        # Enregistrement
        audio_data = sd.rec(
            int(duree_secondes * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Attendre la fin de l'enregistrement
        
        print("✅ Enregistrement terminé !")
        
        # Vérifier que l'audio n'est pas vide
        audio_flat = audio_data.flatten()
        niveau_audio = np.max(np.abs(audio_flat))
        
        if niveau_audio < 0.001:
            print("⚠️  ATTENTION: Niveau audio très faible - vérifiez votre microphone")
        else:
            print(f"📊 Niveau audio détecté: {niveau_audio:.3f}")
        
        return audio_flat
        
    except Exception as e:
        print(f"❌ Erreur enregistrement: {e}")
        return None

async def test_stt_microphone_reel():
    """
    Test STT avec microphone réel et validation humaine obligatoire
    """
    print("\n" + "=" * 60)
    print("🎧 TEST STT MICROPHONE RÉEL - VALIDATION HUMAINE OBLIGATOIRE")
    print("=" * 60)
    
    # Validation GPU obligatoire
    validate_rtx3090_stt()
    
    # Initialiser STT Manager
    print("\n🔧 Initialisation UnifiedSTTManager...")
    stt_manager = UnifiedSTTManager()
    
    # Tests à réaliser
    tests_microphone = [
        {
            "nom": "Test Phrase Courte",
            "duree": 3,
            "instruction": "Dites une phrase courte et claire (ex: 'Bonjour, comment allez-vous ?')"
        },
        {
            "nom": "Test Phrase Longue", 
            "duree": 8,
            "instruction": "Dites une phrase longue avec plusieurs mots (ex: description de votre journée)"
        },
        {
            "nom": "Test Mots Techniques",
            "duree": 5,
            "instruction": "Dites des mots techniques (ex: 'intelligence artificielle', 'apprentissage automatique')"
        }
    ]
    
    resultats_validation = []
    
    for i, test in enumerate(tests_microphone, 1):
        print(f"\n{'='*60}")
        print(f"🎯 TEST {i}/{len(tests_microphone)} : {test['nom']}")
        print(f"📝 Instruction : {test['instruction']}")
        print(f"⏱️ Durée : {test['duree']} secondes")
        print("=" * 60)
        
        # Enregistrement microphone
        audio_data = enregistrer_audio_microphone(test['duree'])
        
        if audio_data is None:
            print(f"❌ Échec enregistrement pour {test['nom']}")
            continue
        
        # Transcription STT
        print(f"\n🤖 TRANSCRIPTION STT EN COURS...")
        start_time = time.time()
        
        try:
            result = await stt_manager.transcribe(audio_data)
            latence_ms = (time.time() - start_time) * 1000
            
            print(f"✅ Transcription terminée en {latence_ms:.0f}ms")
            print(f"📝 Texte transcrit : '{result.text}'")
            print(f"📊 Confiance : {result.confidence:.2f}")
            print(f"⚡ RTF : {result.rtf:.3f}")
            print(f"🎮 Backend utilisé : {result.backend_used}")
            
            # 🎧 VALIDATION HUMAINE OBLIGATOIRE
            print("\n" + "="*60)
            print("🎧 VALIDATION HUMAINE AUDIO OBLIGATOIRE")
            print("="*60)
            
            print("👂 ÉCOUTEZ ATTENTIVEMENT et évaluez la transcription :")
            print(f"   🎤 Ce que vous avez dit : [À compléter]")
            print(f"   🤖 Ce que le STT a transcrit : '{result.text}'")
            print()
            
            # Saisie validation humaine
            phrase_reelle = input("🎤 Tapez exactement ce que vous avez dit : ")
            
            while True:
                precision = input("🎯 Précision transcription (excellent/bon/acceptable/insuffisant) : ").lower()
                if precision in ['excellent', 'bon', 'acceptable', 'insuffisant']:
                    break
                print("❌ Réponse invalide. Utilisez : excellent/bon/acceptable/insuffisant")
            
            while True:
                latence_percue = input("⏱️ Latence perçue (imperceptible/acceptable/gênante) : ").lower()
                if latence_percue in ['imperceptible', 'acceptable', 'gênante']:
                    break
                print("❌ Réponse invalide. Utilisez : imperceptible/acceptable/gênante")
            
            commentaires = input("💬 Commentaires détaillés (optionnel) : ")
            
            # Validation finale
            while True:
                validation = input("🎯 Validation finale (validé/à_corriger/validé_avec_réserves) : ").lower()
                if validation in ['validé', 'à_corriger', 'validé_avec_réserves']:
                    break
                print("❌ Réponse invalide. Utilisez : validé/à_corriger/validé_avec_réserves")
            
            # Calculer précision approximative
            if phrase_reelle.strip():
                mots_reels = phrase_reelle.lower().split()
                mots_transcrits = result.text.lower().split()
                
                # Calcul simple de précision (mots en commun)
                mots_communs = set(mots_reels) & set(mots_transcrits)
                precision_calculee = len(mots_communs) / max(len(mots_reels), 1) * 100
            else:
                precision_calculee = 0
            
            # Enregistrer résultat validation
            resultats_validation.append({
                "test": test['nom'],
                "phrase_reelle": phrase_reelle,
                "texte_transcrit": result.text,
                "latence_ms": latence_ms,
                "rtf": result.rtf,
                "confiance": result.confidence,
                "backend_utilise": result.backend_used,
                "precision_humaine": precision,
                "precision_calculee": precision_calculee,
                "latence_percue": latence_percue,
                "commentaires": commentaires,
                "validation_finale": validation,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"\n✅ Validation {validation.upper()} enregistrée")
            print(f"📊 Précision calculée : {precision_calculee:.1f}%")
            
        except Exception as e:
            print(f"❌ Erreur transcription : {e}")
            resultats_validation.append({
                "test": test['nom'],
                "erreur": str(e),
                "validation_finale": "échec",
                "timestamp": datetime.now().isoformat()
            })
    
    # Rapport final validation humaine
    print("\n" + "="*80)
    print("📊 RAPPORT FINAL VALIDATION HUMAINE MICROPHONE RÉEL")
    print("="*80)
    
    validations_reussies = 0
    precision_moyenne = 0
    latence_moyenne = 0
    
    for i, result in enumerate(resultats_validation, 1):
        print(f"\n🎯 TEST {i} - {result['test']} :")
        if 'erreur' not in result:
            print(f"   🎤 Phrase réelle : '{result['phrase_reelle']}'")
            print(f"   🤖 Transcription : '{result['texte_transcrit']}'")
            print(f"   ⏱️ Latence : {result['latence_ms']:.0f}ms")
            print(f"   📊 Précision calculée : {result['precision_calculee']:.1f}%")
            print(f"   🎯 Précision humaine : {result['precision_humaine']}")
            print(f"   ⏱️ Latence perçue : {result['latence_percue']}")
            print(f"   ✅ Validation : {result['validation_finale']}")
            print(f"   🎮 Backend : {result['backend_utilise']}")
            if result['commentaires']:
                print(f"   💬 Commentaires : {result['commentaires']}")
            
            if result['validation_finale'] == 'validé':
                validations_reussies += 1
            
            precision_moyenne += result['precision_calculee']
            latence_moyenne += result['latence_ms']
        else:
            print(f"   ❌ Erreur : {result['erreur']}")
    
    # Statistiques finales
    if resultats_validation:
        precision_moyenne /= len(resultats_validation)
        latence_moyenne /= len(resultats_validation)
        taux_reussite = (validations_reussies / len(resultats_validation)) * 100
        
        print(f"\n📈 STATISTIQUES FINALES :")
        print(f"   🎯 Taux de réussite : {taux_reussite:.1f}% ({validations_reussies}/{len(resultats_validation)})")
        print(f"   📊 Précision moyenne : {precision_moyenne:.1f}%")
        print(f"   ⏱️ Latence moyenne : {latence_moyenne:.0f}ms")
        
        # Évaluation globale
        if taux_reussite >= 80 and precision_moyenne >= 90:
            print(f"\n🎉 ÉVALUATION GLOBALE : EXCELLENT !")
        elif taux_reussite >= 60 and precision_moyenne >= 80:
            print(f"\n✅ ÉVALUATION GLOBALE : BON")
        elif taux_reussite >= 40 and precision_moyenne >= 70:
            print(f"\n⚠️ ÉVALUATION GLOBALE : ACCEPTABLE")
        else:
            print(f"\n❌ ÉVALUATION GLOBALE : À AMÉLIORER")
    
    # Sauvegarder rapport
    os.makedirs('test_output', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rapport_file = f'test_output/validation_microphone_reel_{timestamp}.json'
    
    with open(rapport_file, 'w', encoding='utf-8') as f:
        json.dump(resultats_validation, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Rapport sauvegardé : {rapport_file}")
    print("\n🎊 VALIDATION HUMAINE MICROPHONE RÉEL TERMINÉE")
    
    return resultats_validation

async def main():
    """Point d'entrée principal"""
    try:
        resultats = await test_stt_microphone_reel()
        print(f"\n✅ Tests microphone terminés avec {len(resultats)} validations")
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors des tests : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 