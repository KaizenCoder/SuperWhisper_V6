#!/usr/bin/env python3
"""
Test de toutes les voix disponibles - SuperWhisper V6
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

# Ajouter le répertoire courant au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import asyncio
from pathlib import Path

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    except ImportError:
        print("⚠️ PyTorch non disponible - validation GPU ignorée")

async def test_coqui_tts():
    """Test Coqui TTS - Voix neurales de haute qualité"""
    print("\n🎤 TEST COQUI TTS - VOIX NEURALES DE HAUTE QUALITÉ")
    print("=" * 60)
    print("🌟 Coqui utilise des voix neurales avancées (Local + RTX 3090)")
    print()
    
    try:
        # Configuration Coqui optimisée RTX 3090
        config = {
            'model_name': 'tts_models/fr/css10/vits',  # Modèle français
            'use_gpu': True,
            'device': 'cuda'  # RTX 3090 forcée
        }
        
        print("1. 🚀 Import Coqui TTS Handler...")
        from TTS.tts_handler_coqui import TTSHandlerCoqui
        
        print("2. 🔧 Initialisation Coqui TTS sur RTX 3090...")
        handler = TTSHandlerCoqui(config)
        
        print("3. 🎯 Test voix Coqui française...")
        test_message = "Bonjour ! Je suis votre assistant LUXA avec une voix neurale Coqui de haute qualité sur RTX 3090."
        
        print(f"   Message: \"{test_message}\"")
        print("   🔊 Synthèse Coqui en cours...")
        
        start_time = time.time()
        handler.speak(test_message)
        synth_time = time.time() - start_time
        
        print(f"   ✅ Synthèse Coqui terminée en {synth_time:.2f}s")
        print("   🌟 Qualité: Neurale haute définition")
        print("   🚀 Accélération: RTX 3090")
        
        return True, "Coqui TTS (Voix neurale française)"
        
    except Exception as e:
        print(f"   ❌ Erreur Coqui TTS: {e}")
        return False, f"Coqui TTS - Erreur: {e}"

async def test_piper_french():
    """Test Piper TTS - Voix française rapide et légère"""
    print("\n🇫🇷 TEST PIPER TTS - VOIX FRANÇAISE RAPIDE")
    print("=" * 60)
    print("⚡ Piper utilise ONNX optimisé RTX 3090 (Ultra-rapide)")
    print()
    
    try:
        # Configuration Piper française
        config = {
            'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
            'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
            'use_gpu': True,
            'sample_rate': 22050
        }
        
        print("1. 🚀 Import Piper French Handler...")
        from TTS.tts_handler_piper_french import TTSHandlerPiperFrench
        
        print("2. 🔧 Initialisation Piper français RTX 3090...")
        handler = TTSHandlerPiperFrench(config)
        
        print("3. 🎯 Test voix Piper française...")
        test_message = "Je suis LUXA avec une voix Piper française optimisée RTX 3090, ultra-rapide et naturelle."
        
        print(f"   Message: \"{test_message}\"")
        print("   🔊 Synthèse Piper en cours...")
        
        start_time = time.time()
        handler.speak(test_message)
        synth_time = time.time() - start_time
        
        print(f"   ✅ Synthèse Piper terminée en {synth_time:.2f}s")
        print("   ⚡ Qualité: ONNX optimisé RTX 3090")
        print("   🇫🇷 Accent: Français natif")
        
        return True, "Piper TTS (Français ONNX RTX 3090)"
        
    except Exception as e:
        print(f"   ❌ Erreur Piper français: {e}")
        return False, f"Piper français - Erreur: {e}"

async def test_piper_espeak():
    """Test Piper eSpeak - Multiple voix disponibles"""
    print("\n🌍 TEST PIPER ESPEAK - MULTIPLES VOIX")
    print("=" * 60)
    print("🗣️ Piper eSpeak offre plusieurs voix européennes")
    print()
    
    try:
        # Configuration Piper eSpeak
        config = {
            'use_gpu': True,
            'sample_rate': 22050,
            'voice': 'fr',  # Voix française
            'speed': 160
        }
        
        print("1. 🚀 Import Piper eSpeak Handler...")
        from TTS.tts_handler_piper_espeak import TTSHandlerPiperEspeak
        
        print("2. 🔧 Initialisation Piper eSpeak RTX 3090...")
        handler = TTSHandlerPiperEspeak(config)
        
        print("3. 🎯 Test voix eSpeak française...")
        test_message = "LUXA avec voix eSpeak française, solution légère et multilingue sur RTX 3090."
        
        print(f"   Message: \"{test_message}\"")
        print("   🔊 Synthèse eSpeak en cours...")
        
        start_time = time.time()
        handler.speak(test_message)
        synth_time = time.time() - start_time
        
        print(f"   ✅ Synthèse eSpeak terminée en {synth_time:.2f}s")
        print("   🌍 Qualité: Multilingue stable")
        print("   ⚡ Performance: RTX 3090 accélérée")
        
        return True, "Piper eSpeak (Multilingue)"
        
    except Exception as e:
        print(f"   ❌ Erreur Piper eSpeak: {e}")
        return False, f"Piper eSpeak - Erreur: {e}"

async def test_alternative_voices():
    """Test menu interactif pour choisir sa voix préférée"""
    print("\n🎭 MENU INTERACTIF - CHOISIR VOTRE VOIX PRÉFÉRÉE")
    print("=" * 60)
    
    # Résultats des tests
    results = []
    
    # Test Coqui (recommandé pour qualité)
    success, info = await test_coqui_tts()
    results.append(("1", "Coqui TTS", info, success, "🌟 RECOMMANDÉ - Qualité maximale"))
    
    # Pause entre tests
    print("\n⏱️ Pause entre les tests...")
    await asyncio.sleep(3)
    
    # Test Piper français (recommandé pour vitesse)
    success, info = await test_piper_french()
    results.append(("2", "Piper Français", info, success, "⚡ RECOMMANDÉ - Ultra-rapide"))
    
    # Pause entre tests
    print("\n⏱️ Pause entre les tests...")
    await asyncio.sleep(3)
    
    # Test Piper eSpeak (recommandé pour polyvalence)
    success, info = await test_piper_espeak()
    results.append(("3", "Piper eSpeak", info, success, "🌍 RECOMMANDÉ - Polyvalent"))
    
    # Afficher le résumé
    print("\n📊 RÉSUMÉ DES VOIX DISPONIBLES")
    print("=" * 60)
    
    for num, name, info, success, recommendation in results:
        status = "✅ DISPONIBLE" if success else "❌ INDISPONIBLE"
        print(f"{num}. {name}")
        print(f"   Status: {status}")
        print(f"   Info: {info}")
        print(f"   {recommendation}")
        print()
    
    # Menu interactif
    print("🎯 MENU INTERACTIF")
    print("Tapez le numéro de la voix que vous préférez pour un test personnalisé :")
    print("1 = Coqui (Qualité max)")
    print("2 = Piper Français (Ultra-rapide)")  
    print("3 = Piper eSpeak (Polyvalent)")
    print("q = Quitter")
    print()
    
    while True:
        try:
            choice = input("🎤 Votre choix (1/2/3/q): ").strip().lower()
            
            if choice == 'q':
                print("👋 Au revoir !")
                break
                
            elif choice == '1':
                print("\n🌟 COQUI TTS - Test personnalisé")
                user_text = input("💬 Votre message pour Coqui: ").strip()
                if user_text:
                    try:
                        config = {'model_name': 'tts_models/fr/css10/vits', 'use_gpu': True}
                        from TTS.tts_handler_coqui import TTSHandlerCoqui
                        handler = TTSHandlerCoqui(config)
                        print("🔊 Coqui dit...")
                        handler.speak(user_text)
                    except Exception as e:
                        print(f"❌ Erreur Coqui: {e}")
                        
            elif choice == '2':
                print("\n⚡ PIPER FRANÇAIS - Test personnalisé")
                user_text = input("💬 Votre message pour Piper: ").strip()
                if user_text:
                    try:
                        config = {
                            'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
                            'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
                            'use_gpu': True
                        }
                        from TTS.tts_handler_piper_french import TTSHandlerPiperFrench
                        handler = TTSHandlerPiperFrench(config)
                        print("🔊 Piper dit...")
                        handler.speak(user_text)
                    except Exception as e:
                        print(f"❌ Erreur Piper: {e}")
                        
            elif choice == '3':
                print("\n🌍 PIPER ESPEAK - Test personnalisé")
                user_text = input("💬 Votre message pour eSpeak: ").strip()
                if user_text:
                    try:
                        config = {'use_gpu': True, 'voice': 'fr', 'speed': 160}
                        from TTS.tts_handler_piper_espeak import TTSHandlerPiperEspeak
                        handler = TTSHandlerPiperEspeak(config)
                        print("🔊 eSpeak dit...")
                        handler.speak(user_text)
                    except Exception as e:
                        print(f"❌ Erreur eSpeak: {e}")
            else:
                print("❌ Choix invalide")
                
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Test interrompu")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    print("🎭 SUPERWHISPER V6 - TEST TOUTES VOIX DISPONIBLES")
    print("🎮 Configuration RTX 3090 (CUDA:1) - Sans SAPI/Hortense")
    print()
    
    # Validation GPU
    validate_rtx3090_configuration()
    
    try:
        success = asyncio.run(test_alternative_voices())
        
        print("\n" + "=" * 60)
        print("🎉 TESTS VOIX TERMINÉS !")
        print("✅ Vous avez maintenant testé toutes les alternatives à SAPI")
        print("🎯 Choisissez la voix qui vous convient le mieux")
        print("🎮 Toutes optimisées pour RTX 3090")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n👋 Tests interrompus")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}") 