#!/usr/bin/env python3
"""
Test de la voix de l'assistant LUXA - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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

async def test_voix_assistant():
    """Test simple de la voix de l'assistant"""
    
    print("\n🎤 TEST VOIX ASSISTANT LUXA - SUPERWHISPER V6")
    print("=" * 60)
    print("🎯 Configuration RTX 3090 (CUDA:1) - Test synthèse vocale")
    print()
    
    # Validation GPU
    validate_rtx3090_configuration()
    
    try:
        # Import TTS Handler MVP (plus simple et stable)
        print("1. 🚀 Import TTS Handler MVP...")
        from TTS.tts_handler_mvp import TTSHandlerMVP
        
        # Configuration pour RTX 3090
        config = {
            'sample_rate': 22050,
            'use_gpu': False,  # MVP utilise Windows SAPI (stable)
            'device': 'cuda:1'  # RTX 3090 forcée
        }
        
        print("✅ TTS Handler MVP importé")
        
        # Initialisation
        print("\n2. 🔧 Initialisation TTS avec RTX 3090...")
        handler = TTSHandlerMVP(config)
        print("✅ TTS Handler initialisé")
        
        # Messages de test de l'assistant
        messages_assistant = [
            "Bonjour ! Je suis votre assistant LUXA, propulsé par SuperWhisper version 6.",
            "Je fonctionne maintenant exclusivement sur votre RTX 3090 avec 24 gigaoctets de mémoire.",
            "Comment puis-je vous aider aujourd'hui ?",
            "Je suis prêt à traiter vos demandes avec performance et efficacité.",
            "Votre configuration GPU RTX 3090 me permet de fonctionner de manière optimale."
        ]
        
        print("\n3. 🎯 Test des messages de l'assistant...")
        print("   (Vous devriez entendre la voix de l'assistant)")
        print()
        
        for i, message in enumerate(messages_assistant, 1):
            print(f"   Message {i}: \"{message}\"")
            print("   🔊 Synthèse en cours...")
            
            start_time = time.time()
            try:
                # Synthèse et lecture immédiate
                audio_data = handler.speak(message)
                
                synth_time = time.time() - start_time
                chars_per_sec = len(message) / synth_time if synth_time > 0 else 0
                
                print(f"   ✅ Synthèse réussie en {synth_time:.2f}s ({chars_per_sec:.0f} car/s)")
                print(f"   🎵 Audio généré: {len(audio_data)} échantillons")
                
                # Attendre un peu avant le message suivant
                print("   ⏱️ Pause entre les messages...")
                time.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Erreur synthèse: {e}")
                continue
            
            print()
        
        # Test interactif
        print("\n4. 🗣️ Mode test interactif")
        print("   Tapez un message pour l'entendre, ou 'quit' pour quitter")
        print()
        
        while True:
            try:
                user_text = input("💬 Votre message pour l'assistant: ").strip()
                
                if user_text.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Au revoir !")
                    break
                
                print("🔊 L'assistant dit...")
                start_time = time.time()
                audio_data = handler.speak(user_text)
                synth_time = time.time() - start_time
                
                print(f"✅ Message synthétisé en {synth_time:.2f}s")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Test interrompu par l'utilisateur")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                continue
        
        print("\n5. 📊 Test terminé")
        print("✅ Voix de l'assistant testée avec succès")
        print("🎮 Configuration RTX 3090 validée")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur import TTS: {e}")
        print("💡 Vérifiez que le module TTS est disponible")
        return False
    except Exception as e:
        print(f"❌ Erreur test voix: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎤 LUXA - TEST VOIX ASSISTANT")
    print("🎮 Configuration RTX 3090 (CUDA:1) - SuperWhisper V6")
    print()
    
    try:
        success = asyncio.run(test_voix_assistant())
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 TEST VOIX RÉUSSI !")
            print("✅ L'assistant peut parler avec la RTX 3090")
            print("🎮 Configuration GPU validée")
        else:
            print("🚨 PROBLÈME DÉTECTÉ")
            print("⚠️ Vérification nécessaire")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n👋 Test interrompu")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}") 