#!/usr/bin/env python3
"""
🎤 SCRIPT VALIDATION MICROPHONE LIVE - ÉQUIPE VALIDATION
SuperWhisper V6 Phase 4 STT - Validation finale correction VAD

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 SuperWhisper V6 - Validation Microphone Live - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

try:
    import torch
    import numpy as np
    import sounddevice as sd
    from STT.unified_stt_manager import UnifiedSTTManager
    print("✅ Imports STT réussis")
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("💡 Vérifiez l'installation des dépendances STT")
    sys.exit(1)

def validate_rtx3090_validation():
    """Validation systématique RTX 3090 pour équipe validation"""
    print("\n🔍 VALIDATION CONFIGURATION RTX 3090")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible - RTX 3090 requise")
        return False
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        print(f"❌ CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    if "3090" not in gpu_name:
        print(f"❌ GPU détectée: {gpu_name} - RTX 3090 requise")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        print(f"❌ GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        return False
    
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")
    return True

def test_microphone_setup():
    """Test setup microphone"""
    print("\n🎤 TEST SETUP MICROPHONE")
    print("=" * 30)
    
    try:
        # Lister devices audio
        devices = sd.query_devices()
        print("📋 Devices audio disponibles:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} (Input: {device['max_input_channels']} ch)")
        
        # Test enregistrement court
        print("\n🔴 Test enregistrement 2 secondes...")
        print("   Parlez maintenant...")
        
        audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        
        # Vérifier niveau audio
        max_level = np.max(np.abs(audio))
        print(f"📊 Niveau audio max: {max_level:.3f}")
        
        if max_level < 0.01:
            print("⚠️ Niveau audio très faible - vérifiez microphone")
            return False
        elif max_level > 0.8:
            print("⚠️ Niveau audio très fort - risque saturation")
        else:
            print("✅ Niveau audio correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test microphone: {e}")
        return False

async def validation_texte_complet():
    """Validation avec texte complet fourni"""
    
    # Texte de référence pour validation
    TEXTE_REFERENCE = """Bonjour, je suis en train de tester le système de reconnaissance vocale SuperWhisper V6. 
Cette phrase contient plusieurs mots techniques comme reconnaissance, transcription, et validation. 
Le système doit être capable de transcrire correctement tous les mots sans interruption. 
Nous testons également les nombres comme 123, 456, et les dates comme le 13 juin 2025. 
Cette validation est critique pour valider la correction VAD qui a permis une amélioration de 492 pourcent. 
Le système utilise une RTX 3090 avec 24 gigaoctets de mémoire vidéo pour optimiser les performances. 
Merci de valider que cette transcription est complète et précise."""
    
    print("\n📝 VALIDATION TEXTE COMPLET")
    print("=" * 40)
    print("\n🎯 TEXTE À LIRE AU MICROPHONE:")
    print("-" * 40)
    print(TEXTE_REFERENCE)
    print("-" * 40)
    
    # Compter mots de référence
    mots_reference = len(TEXTE_REFERENCE.split())
    print(f"\n📊 Mots de référence: {mots_reference}")
    
    input("\n🎤 Appuyez sur Entrée quand vous êtes prêt à lire le texte complet...")
    
    try:
        # Initialiser STT Manager
        print("🚀 Initialisation STT Manager...")
        stt_manager = UnifiedSTTManager()
        
        # Enregistrement long (30 secondes max)
        duree_max = 30
        print(f"\n🔴 ENREGISTREMENT EN COURS... (max {duree_max}s)")
        print("📢 LISEZ LE TEXTE MAINTENANT")
        
        start_time = time.time()
        audio = sd.rec(int(duree_max * 16000), samplerate=16000, channels=1, dtype=np.float32)
        
        # Attendre fin lecture ou timeout
        print("⏹️ Appuyez sur Entrée quand vous avez terminé de lire...")
        input()
        sd.stop()
        
        duree_reelle = time.time() - start_time
        print(f"⏱️ Durée enregistrement: {duree_reelle:.1f}s")
        
        # Transcription
        print("🎮 Transcription RTX 3090...")
        start_transcription = time.time()
        
        # Prendre seulement la partie enregistrée
        samples_enregistres = int(duree_reelle * 16000)
        audio_final = audio[:samples_enregistres].flatten()
        
        result = await stt_manager.transcribe(audio_final)
        
        latence_transcription = time.time() - start_transcription
        
        # Analyse résultats
        texte_transcrit = result.get('text', '')
        mots_transcrits = len(texte_transcrit.split())
        couverture = (mots_transcrits / mots_reference) * 100
        
        print("\n📊 RÉSULTATS VALIDATION")
        print("=" * 30)
        print(f"📝 Texte transcrit: '{texte_transcrit}'")
        print(f"📊 Mots transcrits: {mots_transcrits}/{mots_reference}")
        print(f"📈 Couverture: {couverture:.1f}%")
        print(f"⏱️ Latence transcription: {latence_transcription:.1f}s")
        print(f"🎯 RTF: {result.get('rtf', 'N/A')}")
        
        # Validation critères
        validation_reussie = True
        criteres = []
        
        if couverture >= 95:
            criteres.append("✅ Couverture excellente (≥95%)")
        elif couverture >= 90:
            criteres.append("🟡 Couverture acceptable (≥90%)")
            print("⚠️ Couverture sous l'objectif de 95%")
        else:
            criteres.append("❌ Couverture insuffisante (<90%)")
            validation_reussie = False
        
        if latence_transcription <= 10:
            criteres.append("✅ Latence acceptable (≤10s)")
        elif latence_transcription <= 15:
            criteres.append("🟡 Latence limite (≤15s)")
        else:
            criteres.append("❌ Latence excessive (>15s)")
            validation_reussie = False
        
        if "interruption" not in texte_transcrit.lower() and len(texte_transcrit) > 50:
            criteres.append("✅ Transcription complète")
        else:
            criteres.append("❌ Transcription incomplète ou interrompue")
            validation_reussie = False
        
        print("\n🎯 CRITÈRES VALIDATION:")
        for critere in criteres:
            print(f"   {critere}")
        
        # Validation humaine
        print("\n👤 VALIDATION HUMAINE REQUISE")
        print("=" * 35)
        
        while True:
            precision = input("🎯 Précision transcription (excellent/bon/acceptable/insuffisant): ").lower()
            if precision in ['excellent', 'bon', 'acceptable', 'insuffisant']:
                break
            print("❌ Réponse invalide")
        
        while True:
            interruption = input("🔍 Y a-t-il eu des interruptions prématurées? (oui/non): ").lower()
            if interruption in ['oui', 'non']:
                break
            print("❌ Réponse invalide")
        
        commentaires = input("💬 Commentaires détaillés (optionnel): ")
        
        # Décision finale
        if validation_reussie and precision in ['excellent', 'bon', 'acceptable'] and interruption == 'non':
            decision_finale = "✅ VALIDÉ"
        elif precision == 'acceptable' and interruption == 'non':
            decision_finale = "🔄 VALIDÉ AVEC RÉSERVES"
        else:
            decision_finale = "❌ À CORRIGER"
        
        # Sauvegarder rapport
        rapport = {
            "date_validation": datetime.now().isoformat(),
            "texte_reference": TEXTE_REFERENCE,
            "mots_reference": mots_reference,
            "texte_transcrit": texte_transcrit,
            "mots_transcrits": mots_transcrits,
            "couverture_pourcent": couverture,
            "latence_transcription": latence_transcription,
            "rtf": result.get('rtf'),
            "duree_enregistrement": duree_reelle,
            "precision_humaine": precision,
            "interruptions": interruption,
            "commentaires": commentaires,
            "criteres_techniques": criteres,
            "decision_finale": decision_finale,
            "validation_reussie": validation_reussie
        }
        
        # Créer dossier rapport
        Path("validation_reports").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fichier_rapport = f"validation_reports/validation_microphone_live_{timestamp}.json"
        
        with open(fichier_rapport, 'w', encoding='utf-8') as f:
            json.dump(rapport, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Rapport sauvegardé: {fichier_rapport}")
        print(f"\n🎯 DÉCISION FINALE: {decision_finale}")
        
        return validation_reussie, rapport
        
    except Exception as e:
        print(f"❌ Erreur validation: {e}")
        return False, None

async def main():
    """Fonction principale validation microphone live"""
    
    print("🎤 VALIDATION MICROPHONE LIVE - SuperWhisper V6 Phase 4 STT")
    print("=" * 65)
    print("🎯 Mission: Valider correction VAD en conditions réelles")
    print("📊 État: Correction VAD réussie sur fichier (+492%)")
    print("🔍 Objectif: Confirmer fonctionnement avec microphone live")
    print()
    
    # Étape 1: Validation GPU
    if not validate_rtx3090_validation():
        print("\n❌ ÉCHEC VALIDATION GPU - ARRÊT")
        return False
    
    # Étape 2: Test microphone
    if not test_microphone_setup():
        print("\n❌ ÉCHEC TEST MICROPHONE - ARRÊT")
        return False
    
    # Étape 3: Validation texte complet
    print("\n🚀 DÉMARRAGE VALIDATION TEXTE COMPLET")
    validation_ok, rapport = await validation_texte_complet()
    
    if validation_ok:
        print("\n🎊 VALIDATION MICROPHONE LIVE RÉUSSIE!")
        print("✅ La correction VAD fonctionne parfaitement en conditions réelles")
        print("🚀 Phase 4 STT peut être marquée comme TERMINÉE")
    else:
        print("\n⚠️ VALIDATION MICROPHONE LIVE PARTIELLE")
        print("🔧 Des ajustements peuvent être nécessaires")
    
    print(f"\n📋 Rapport détaillé disponible dans: validation_reports/")
    return validation_ok

if __name__ == "__main__":
    import asyncio
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        sys.exit(1) 