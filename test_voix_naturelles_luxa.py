#!/usr/bin/env python3
"""
TEST VOIX NATURELLES LUXA - Voix neurales de qualité
🚨 RTX 3090 (CUDA:1) - VOIX NATURELLES GARANTIES
"""

import os
import sys

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU RTX 3090 configurée")

def test_piper_naturel():
    """Test voix Piper naturelle (fr_FR-siwis-medium)"""
    
    print("\n🎭 TEST PIPER VOIX NATURELLE")
    print("=" * 40)
    
    try:
        sys.path.append('TTS')
        from tts_handler_piper_french import TTSHandlerPiperFrench
        
        # Configuration CORRECTE d'après les docs
        config = {
            'model_path': 'models/fr_FR-siwis-medium.onnx',  # BON CHEMIN
            'use_gpu': True,
            'sample_rate': 22050
        }
        
        print(f"📄 Modèle Piper: {config['model_path']}")
        
        # Vérification fichier
        if not os.path.exists(config['model_path']):
            print("❌ Modèle Piper manquant dans models/")
            # Tentative chemin alternatif
            alt_path = 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx'
            if os.path.exists(alt_path):
                config['model_path'] = alt_path
                config['config_path'] = alt_path + '.json'
                print(f"✅ Modèle trouvé: {alt_path}")
            else:
                print("❌ Aucun modèle Piper trouvé")
                return False
        
        print("1. 🚀 Initialisation Piper naturel...")
        handler = TTSHandlerPiperFrench(config)
        print("✅ Piper initialisé")
        
        # Test voix naturelle
        texte = "Bonjour ! Je suis LUXA avec une voix naturelle Piper."
        print(f"2. 🗣️ Test Piper: '{texte}'")
        
        print("3. 🔊 Synthèse Piper naturelle...")
        handler.speak(texte)
        print("✅ Piper naturel terminé")
        
        print("🎭 VOIX PIPER: Plus naturelle qu'Hortense ?")
        return True
        
    except ImportError:
        print("❌ Handler Piper French non disponible")
        return test_piper_fixed()
    except Exception as e:
        print(f"❌ Erreur Piper: {e}")
        return test_piper_fixed()

def test_piper_fixed():
    """Test avec Piper Fixed handler"""
    
    try:
        print("\n🔧 Test Piper Fixed...")
        from tts_handler_piper_fixed import TTSHandlerPiperFixed
        
        config = {
            'model_path': 'models/fr_FR-siwis-medium.onnx',
            'use_gpu': True
        }
        
        # Vérification chemin alternatif si nécessaire
        if not os.path.exists(config['model_path']):
            alt_path = 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx'
            if os.path.exists(alt_path):
                config['model_path'] = alt_path
                config['config_path'] = alt_path + '.json'
        
        handler = TTSHandlerPiperFixed(config)
        
        texte = "LUXA utilise maintenant Piper pour une voix plus naturelle."
        print(f"🗣️ Piper Fixed: '{texte}'")
        
        handler.speak(texte)
        print("✅ Piper Fixed terminé")
        
        return True
        
    except Exception as e:
        print(f"❌ Piper Fixed échoué: {e}")
        return False

def test_coqui_naturel():
    """Test voix Coqui (la plus naturelle)"""
    
    print("\n🌟 TEST COQUI VOIX ULTRA-NATURELLE")
    print("=" * 40)
    
    try:
        from tts_handler_coqui import TTSHandlerCoqui
        
        config = {
            'model_name': 'tts_models/fr/css10/vits',  # Modèle français Coqui
            'use_gpu': True,
            'sample_rate': 22050
        }
        
        print("1. 🚀 Initialisation Coqui neural...")
        handler = TTSHandlerCoqui(config)
        print("✅ Coqui initialisé")
        
        # Test voix ultra-naturelle
        texte = "Bonjour ! Je suis LUXA avec Coqui, la voix la plus naturelle."
        print(f"2. 🗣️ Test Coqui: '{texte}'")
        
        print("3. 🔊 Synthèse Coqui ultra-naturelle...")
        handler.speak(texte)
        print("✅ Coqui terminé")
        
        print("🌟 VOIX COQUI: La plus naturelle de toutes !")
        return True
        
    except ImportError:
        print("❌ Coqui TTS non installé")
        print("💡 Installez avec: pip install TTS")
        return False
    except Exception as e:
        print(f"❌ Erreur Coqui: {e}")
        return False

def test_voix_disponibles():
    """Liste toutes les voix disponibles"""
    
    print("\n📋 VOIX DISPONIBLES LUXA")
    print("=" * 30)
    
    voix_testees = []
    
    # Test Hortense (déjà validée)
    print("✅ Microsoft Hortense: FONCTIONNE (moins naturelle)")
    voix_testees.append("Hortense")
    
    # Test Piper
    if test_piper_naturel():
        print("✅ Piper Neural: FONCTIONNE (naturelle)")
        voix_testees.append("Piper")
    else:
        print("❌ Piper Neural: Non disponible")
    
    # Test Coqui  
    if test_coqui_naturel():
        print("✅ Coqui Neural: FONCTIONNE (ultra-naturelle)")
        voix_testees.append("Coqui")
    else:
        print("❌ Coqui Neural: Non disponible")
    
    return voix_testees

if __name__ == "__main__":
    print("🎭 TEST VOIX NATURELLES LUXA")
    print("🎯 Recherche voix plus naturelles qu'Hortense")
    print()
    
    voix_ok = test_voix_disponibles()
    
    print(f"\n📊 RÉSULTATS:")
    print(f"✅ Voix fonctionnelles: {len(voix_ok)}")
    print(f"🎭 Voix disponibles: {', '.join(voix_ok)}")
    
    if len(voix_ok) > 1:
        print("\n🎉 CHOIX MULTIPLES: Sélectionnez votre voix préférée!")
    else:
        print("\n💡 Installation voix supplémentaires recommandée")
    
    print("\n🎤 Quelle voix préférez-vous pour LUXA ?") 