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

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Test setup microphone avec sélection automatique RODE NT-USB"""
    print("\n🎤 TEST SETUP MICROPHONE")
    print("=" * 30)
    
    try:
        # Lister devices audio
        devices = sd.query_devices()
        print("📋 Devices audio disponibles:")
        
        # Chercher TOUS les microphones RODE NT-USB
        rode_devices = []
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} (Input: {device['max_input_channels']} ch)")
                input_devices.append((i, device['name']))
                
                # Détecter TOUTES les instances RODE NT-USB
                if "RODE NT-USB" in device['name']:
                    rode_devices.append(i)
                    print(f"   🎯 RODE NT-USB détecté: Device {i}")
        
        # Tester chaque instance RODE NT-USB pour trouver celle qui fonctionne
        selected_device = None
        
        if rode_devices:
            print(f"\n🔍 Test de {len(rode_devices)} instances RODE NT-USB...")
            
            for device_id in rode_devices:
                print(f"\n🧪 Test Device {device_id}...")
                try:
                    # Test rapide 1 seconde
                    test_audio = sd.rec(int(1 * 16000), samplerate=16000, channels=1, dtype=np.float32, device=device_id)
                    sd.wait()
                    
                    # Vérifier si l'enregistrement a fonctionné
                    max_level = np.max(np.abs(test_audio))
                    if max_level > 0.001:  # Seuil très bas pour détecter activité
                        print(f"✅ Device {device_id} fonctionnel (niveau: {max_level:.6f})")
                        selected_device = device_id
                        break
                    else:
                        print(f"⚠️ Device {device_id} silencieux (niveau: {max_level:.6f})")
                        
                except Exception as e:
                    print(f"❌ Device {device_id} erreur: {e}")
                    continue
            
            if selected_device is None:
                print("⚠️ Aucune instance RODE NT-USB fonctionnelle trouvée")
                # Fallback sur le premier device RODE trouvé
                selected_device = rode_devices[0]
                print(f"🔄 Utilisation Device {selected_device} par défaut")
            else:
                print(f"\n✅ Sélection automatique: RODE NT-USB (Device {selected_device})")
                
        else:
            print(f"\n⚠️ RODE NT-USB non trouvé, sélection manuelle requise")
            print("📋 Microphones d'entrée disponibles:")
            for i, (device_id, name) in enumerate(input_devices):
                print(f"   {i}: Device {device_id} - {name}")
            
            while True:
                try:
                    choice = int(input("🎯 Sélectionnez le numéro du microphone à utiliser: "))
                    if 0 <= choice < len(input_devices):
                        selected_device = input_devices[choice][0]
                        break
                    else:
                        print("❌ Numéro invalide")
                except ValueError:
                    print("❌ Veuillez entrer un numéro")
        
        print(f"🎤 Microphone sélectionné: Device {selected_device}")
        
        # Test enregistrement final avec microphone sélectionné
        print(f"\n🔴 Test enregistrement 3 secondes avec Device {selected_device}...")
        print("   Parlez fort et clairement maintenant...")
        
        # Enregistrement avec device spécifique
        audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype=np.float32, device=selected_device)
        sd.wait()
        
        # Vérifier niveau audio
        max_level = np.max(np.abs(audio))
        rms_level = np.sqrt(np.mean(audio**2))
        
        print(f"📊 Niveau audio max: {max_level:.3f}")
        print(f"📊 Niveau RMS: {rms_level:.3f}")
        
        if max_level < 0.01:
            print("⚠️ Niveau audio très faible")
            print("💡 Suggestions:")
            print("   - Vérifiez que le microphone est branché")
            print("   - Augmentez le volume du microphone dans Windows")
            print("   - Rapprochez-vous du microphone")
            print("   - Parlez plus fort")
            
            retry = input("🔄 Voulez-vous réessayer avec un autre microphone? (oui/non/continuer): ").lower()
            if retry == 'oui':
                return test_microphone_setup()  # Récursion pour réessayer
            elif retry == 'continuer':
                print("⚠️ Continuation avec niveau audio faible - résultats peuvent être affectés")
                return True, selected_device
            else:
                return False, None
                
        elif max_level > 0.8:
            print("⚠️ Niveau audio très fort - risque saturation")
            print("💡 Réduisez le volume du microphone")
        else:
            print("✅ Niveau audio correct")
        
        # Retourner le device sélectionné pour utilisation ultérieure
        return True, selected_device
        
    except Exception as e:
        print(f"❌ Erreur test microphone: {e}")
        return False, None

async def validation_texte_complet(selected_device=None):
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
    
    if selected_device is not None:
        print(f"🎤 Microphone sélectionné: Device {selected_device}")
    
    input("\n🎤 Appuyez sur Entrée quand vous êtes prêt à lire le texte complet...")
    
    try:
        # Initialiser STT Manager
        print("🚀 Initialisation STT Manager...")
        
        # Configuration pour validation microphone live
        config = {
            'timeout_per_minute': 10.0,  # Timeout généreux pour validation
            'max_retries': 3,
            'cache_enabled': True,
            'circuit_breaker_enabled': True,
            'fallback_chain': ['prism_primary'],  # CRITIQUE: définir l'ordre des backends
            'backends': [
                {
                    'name': 'prism_primary',
                    'type': 'prism',
                    'model': 'large-v2',
                    'compute_type': 'float16',
                    'language': 'fr',
                    'beam_size': 5,
                    'vad_filter': True
                }
            ]
        }
        
        stt_manager = UnifiedSTTManager(config=config)
        
        # Enregistrement long (30 secondes max)
        duree_max = 30
        print(f"\n🔴 ENREGISTREMENT EN COURS... (max {duree_max}s)")
        print("📢 LISEZ LE TEXTE MAINTENANT")
        
        start_time = time.time()
        
        # Utiliser le device sélectionné si disponible
        if selected_device is not None:
            audio = sd.rec(int(duree_max * 16000), samplerate=16000, channels=1, dtype=np.float32, device=selected_device)
        else:
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
        
        try:
            # Prendre seulement la partie enregistrée
            samples_enregistres = int(duree_reelle * 16000)
            audio_data = audio[:samples_enregistres].flatten()
            
            print(f"🔍 Debug - Audio shape: {audio_data.shape}")
            print(f"🔍 Debug - Audio dtype: {audio_data.dtype}")
            print(f"🔍 Debug - Audio min/max: {audio_data.min():.6f}/{audio_data.max():.6f}")
            
            result = await stt_manager.transcribe(audio_data)
            latence_transcription = time.time() - start_transcription
            
            print(f"🔍 Debug - Result type: {type(result)}")
            print(f"🔍 Debug - Result success: {getattr(result, 'success', 'N/A')}")
            print(f"🔍 Debug - Result error: {getattr(result, 'error', 'N/A')}")
            print(f"🔍 Debug - Result backend: {getattr(result, 'backend_used', 'N/A')}")
            print(f"🔍 Debug - Result text length: {len(getattr(result, 'text', ''))}")
            
        except Exception as e:
            print(f"❌ Erreur transcription: {e}")
            print(f"❌ Type erreur: {type(e)}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return False
        
        # Analyse résultats
        texte_transcrit = result.text if hasattr(result, 'text') else result.get('text', '')
        mots_transcrits = len(texte_transcrit.split())
        couverture = (mots_transcrits / mots_reference) * 100
        
        print("\n📊 RÉSULTATS VALIDATION")
        print("=" * 30)
        print(f"📝 Texte transcrit: '{texte_transcrit}'")
        print(f"📊 Mots transcrits: {mots_transcrits}/{mots_reference}")
        print(f"📈 Couverture: {couverture:.1f}%")
        print(f"⏱️ Latence transcription: {latence_transcription:.1f}s")
        print(f"🎯 RTF: {result.rtf if hasattr(result, 'rtf') else 'N/A'}")
        print(f"�� Backend utilisé: {result.backend_used if hasattr(result, 'backend_used') else 'N/A'}")
        print(f"✅ Succès: {result.success if hasattr(result, 'success') else 'N/A'}")
        
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
            "rtf": result.rtf if hasattr(result, 'rtf') else None,
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
    validation_ok, selected_device = test_microphone_setup()
    
    if not validation_ok:
        print("\n❌ ÉCHEC TEST MICROPHONE - ARRÊT")
        return False
    
    # Étape 3: Validation texte complet
    print("\n🚀 DÉMARRAGE VALIDATION TEXTE COMPLET")
    validation_ok, rapport = await validation_texte_complet(selected_device)
    
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