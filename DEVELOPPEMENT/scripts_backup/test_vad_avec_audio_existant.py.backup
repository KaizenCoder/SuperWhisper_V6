#!/usr/bin/env python3
"""
Test Correction VAD avec Fichier Audio Existant - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")

import torch
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

def validate_rtx3090():
    """Validation GPU RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

def test_vad_avec_audio_reel():
    """Test correction VAD avec fichier audio réel"""
    
    print("\n" + "="*60)
    print("🔧 TEST CORRECTION VAD AVEC AUDIO RÉEL")
    print("="*60)
    
    validate_rtx3090()
    
    # Import backend
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
    
    print("🤖 Initialisation Prism STT Backend...")
    start_init = time.time()
    backend = PrismSTTBackend(config)
    init_time = time.time() - start_init
    print(f"✅ Backend initialisé en {init_time:.2f}s")
    
    # Fichiers audio à tester
    fichiers_test = [
        {
            "nom": "Validation Utilisateur Complet",
            "fichier": "test_output/validation_utilisateur_complet.wav",
            "description": "Fichier de validation utilisateur complet"
        },
        {
            "nom": "Demo Batch Long",
            "fichier": "test_output/demo_batch_long_20250612_151733.wav",
            "description": "Fichier demo batch long"
        },
        {
            "nom": "Demo Batch Moyen",
            "fichier": "test_output/demo_batch_moyen_20250612_151733.wav",
            "description": "Fichier demo batch moyen"
        }
    ]
    
    resultats_tests = []
    
    for i, test_audio in enumerate(fichiers_test, 1):
        print(f"\n🎯 TEST {i}/{len(fichiers_test)}: {test_audio['nom']}")
        print(f"📁 Fichier: {test_audio['fichier']}")
        print(f"📝 Description: {test_audio['description']}")
        
        fichier_path = Path(test_audio['fichier'])
        
        if not fichier_path.exists():
            print(f"❌ Fichier non trouvé: {fichier_path}")
            resultats_tests.append({
                "test": test_audio['nom'],
                "erreur": "Fichier non trouvé",
                "timestamp": datetime.now().isoformat()
            })
            continue
        
        try:
            # Charger audio
            print(f"📂 Chargement audio...")
            audio_data, sample_rate = sf.read(str(fichier_path))
            
            # Convertir vers format backend
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Mono
            
            if sample_rate != 16000:
                print(f"🔄 Resampling de {sample_rate}Hz vers 16000Hz...")
                import resampy
                audio_data = resampy.resample(audio_data, sample_rate, 16000)
            
            audio_data = audio_data.astype(np.float32)
            duree_audio = len(audio_data) / 16000
            
            print(f"🎵 Audio chargé: {len(audio_data)} samples, {duree_audio:.1f}s")
            
            # Test transcription avec VAD corrigé
            print(f"🔧 TRANSCRIPTION AVEC VAD CORRIGÉ...")
            start_time = time.time()
            
            import asyncio
            result = asyncio.run(backend.transcribe(audio_data))
            
            duration_ms = (time.time() - start_time) * 1000
            
            if result.success:
                texte_transcrit = result.text.strip()
                mots_transcrits = len(texte_transcrit.split()) if texte_transcrit else 0
                
                print(f"✅ TRANSCRIPTION RÉUSSIE en {duration_ms:.0f}ms")
                print(f"📝 Texte transcrit ({mots_transcrits} mots):")
                print(f"   '{texte_transcrit}'")
                print(f"📊 Métriques:")
                print(f"   - Durée audio: {duree_audio:.1f}s")
                print(f"   - Mots transcrits: {mots_transcrits}")
                print(f"   - Confiance: {result.confidence:.3f}")
                print(f"   - RTF: {result.rtf:.3f}")
                print(f"   - Latence: {duration_ms:.0f}ms")
                print(f"   - Backend: {result.backend_used}")
                
                # Analyser résultat
                if mots_transcrits > 0:
                    if duration_ms < 1000:
                        performance = "EXCELLENTE"
                    elif duration_ms < 2000:
                        performance = "BONNE"
                    else:
                        performance = "ACCEPTABLE"
                    
                    print(f"🎉 SUCCÈS: {mots_transcrits} mots transcrits - Performance {performance}")
                    
                    # Vérifier si c'est une amélioration par rapport au problème précédent
                    if mots_transcrits > 25:  # Plus que les 25 mots du problème initial
                        print(f"🚀 AMÉLIORATION CONFIRMÉE!")
                        print(f"   Avant correction: 25 mots maximum")
                        print(f"   Après correction: {mots_transcrits} mots")
                        statut = "AMÉLIORATION"
                    else:
                        print(f"⚠️ Transcription partielle: {mots_transcrits} mots")
                        statut = "PARTIEL"
                else:
                    print(f"❌ AUCUN MOT TRANSCRIT")
                    statut = "ÉCHEC"
                
                resultats_tests.append({
                    "test": test_audio['nom'],
                    "fichier": test_audio['fichier'],
                    "duree_audio": duree_audio,
                    "texte_transcrit": texte_transcrit,
                    "mots_transcrits": mots_transcrits,
                    "latence_ms": duration_ms,
                    "rtf": result.rtf,
                    "confiance": result.confidence,
                    "backend_used": result.backend_used,
                    "statut": statut,
                    "performance": performance if mots_transcrits > 0 else "N/A",
                    "timestamp": datetime.now().isoformat()
                })
                
            else:
                print(f"❌ ÉCHEC TRANSCRIPTION: {result.error}")
                resultats_tests.append({
                    "test": test_audio['nom'],
                    "fichier": test_audio['fichier'],
                    "erreur": result.error,
                    "statut": "ERREUR",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            print(f"❌ ERREUR TEST: {e}")
            import traceback
            traceback.print_exc()
            resultats_tests.append({
                "test": test_audio['nom'],
                "fichier": test_audio['fichier'],
                "erreur": str(e),
                "statut": "EXCEPTION",
                "timestamp": datetime.now().isoformat()
            })
    
    # Rapport final
    print("\n" + "="*60)
    print("📊 RAPPORT FINAL - TEST CORRECTION VAD AVEC AUDIO RÉEL")
    print("="*60)
    
    succes = 0
    ameliorations = 0
    echecs = 0
    
    for result in resultats_tests:
        print(f"\n🎯 {result['test']}:")
        if 'erreur' not in result:
            print(f"   📁 Fichier: {result['fichier']}")
            print(f"   ⏱️ Durée: {result['duree_audio']:.1f}s")
            print(f"   📝 Mots: {result['mots_transcrits']}")
            print(f"   ⚡ Latence: {result['latence_ms']:.0f}ms")
            print(f"   📊 RTF: {result['rtf']:.3f}")
            print(f"   🎯 Statut: {result['statut']}")
            
            if result['statut'] == 'AMÉLIORATION':
                ameliorations += 1
            elif result['mots_transcrits'] > 0:
                succes += 1
            else:
                echecs += 1
        else:
            print(f"   ❌ Erreur: {result['erreur']}")
            echecs += 1
    
    print(f"\n📈 RÉSUMÉ GLOBAL:")
    print(f"   🚀 Améliorations: {ameliorations}")
    print(f"   ✅ Succès: {succes}")
    print(f"   ❌ Échecs: {echecs}")
    print(f"   📊 Total: {len(resultats_tests)}")
    
    # Sauvegarder rapport
    os.makedirs('test_output', exist_ok=True)
    rapport_file = f'test_output/test_vad_audio_reel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(rapport_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "correction_vad": "APPLIQUÉE",
            "tests_realises": len(resultats_tests),
            "ameliorations": ameliorations,
            "succes": succes,
            "echecs": echecs,
            "resultats": resultats_tests
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Rapport sauvegardé: {rapport_file}")
    
    # Conclusion
    if ameliorations > 0:
        print(f"\n🎉 CORRECTION VAD EFFICACE!")
        print(f"   {ameliorations} amélioration(s) détectée(s)")
        print(f"   La correction VAD fonctionne correctement")
        return True
    elif succes > 0:
        print(f"\n✅ CORRECTION VAD FONCTIONNELLE")
        print(f"   {succes} transcription(s) réussie(s)")
        print(f"   Pas d'amélioration majeure mais fonctionnel")
        return True
    else:
        print(f"\n❌ CORRECTION VAD INSUFFISANTE")
        print(f"   Aucune transcription réussie")
        print(f"   Investigation supplémentaire requise")
        return False

def main():
    """Test principal"""
    print("🔧 TEST CORRECTION VAD AVEC FICHIERS AUDIO RÉELS")
    print("="*60)
    
    try:
        success = test_vad_avec_audio_reel()
        
        print("\n" + "="*60)
        if success:
            print("🎉 TEST RÉUSSI: Correction VAD validée avec audio réel!")
            print("📋 Actions suggérées:")
            print("   1. Analyser rapport détaillé généré")
            print("   2. Tester avec d'autres fichiers audio si nécessaire")
            print("   3. Procéder aux tests de validation humaine")
            print("   4. Documenter les résultats dans le journal")
        else:
            print("❌ TEST ÉCHOUÉ: Correction VAD insuffisante")
            print("📋 Actions requises:")
            print("   1. Analyser les erreurs dans le rapport")
            print("   2. Vérifier les paramètres VAD")
            print("   3. Tester avec des fichiers audio plus simples")
            print("   4. Contacter expert si problème persiste")
        print("="*60)
        
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 