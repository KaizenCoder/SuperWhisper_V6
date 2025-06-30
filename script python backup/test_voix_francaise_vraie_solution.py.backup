#!/usr/bin/env python3
"""
TEST VOIX FRANÇAISE VRAIE SOLUTION - LUXA SuperWhisper V6
🚨 UTILISE LA VRAIE CONFIG DOCUMENTÉE QUI MARCHE
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

def test_vraie_solution_francaise():
    """Test avec la VRAIE configuration trouvée dans la documentation"""
    
    print("\n🎭 TEST VRAIE SOLUTION VOIX FRANÇAISE")
    print("=" * 50)
    print("📋 Configuration d'après DEBUG_TTS_FRENCH_VOICE_ISSUE.md")
    print("✅ Utilise mvp_settings.yaml: models/fr_FR-siwis-medium.onnx")
    
    try:
        # 1. Configuration CORRECTE selon documentation
        config = {
            'model_path': 'models/fr_FR-siwis-medium.onnx',  # ✅ BON CHEMIN
            'use_gpu': True,
            'sample_rate': 22050
        }
        
        print(f"1. 📁 Modèle: {config['model_path']}")
        
        # 2. Vérifier si fichier existe
        import os
        if os.path.exists(config['model_path']):
            print("✅ Fichier modèle trouvé")
        else:
            print(f"❌ Fichier modèle introuvable: {config['model_path']}")
            print("🔍 Recherche autres emplacements...")
            
            # Recherche alternative
            possible_paths = [
                'models/fr_FR-siwis-medium.onnx',
                'TTS/models/fr_FR-siwis-medium.onnx',
                'D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx',
                'piper/models/fr_FR-siwis-medium.onnx'
            ]
            
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    print(f"✅ Trouvé: {path}")
                    break
                else:
                    print(f"❌ Pas trouvé: {path}")
            
            if found_path:
                config['model_path'] = found_path
            else:
                print("❌ AUCUN modèle français trouvé - fallback SAPI")
                test_sapi_francais()
                return
        
        # 3. Test avec handler Piper
        print("2. 🚀 Initialisation Piper avec VRAIE config...")
        
        sys.path.append('TTS')
        from tts_handler_piper_french import TTSHandlerPiperFrench
        
        handler = TTSHandlerPiperFrench(config)
        print("✅ Handler Piper initialisé")
        
        # 4. Test avec phrase française LONGUE
        texte_long = """
        Bonjour, je suis LUXA, votre assistant vocal intelligent. 
        Je utilise désormais une voix française naturelle de haute qualité.
        Cette voix est générée par un modèle neuronal Piper optimisé pour le français.
        Vous pouvez maintenant converser avec moi dans votre langue maternelle.
        Comment puis-je vous aider aujourd'hui ?
        """
        
        print(f"3. 🗣️ Texte long ({len(texte_long)} caractères):")
        print(f"   '{texte_long[:50]}...'")
        
        # 5. Synthèse
        print("4. 🎵 Synthèse audio...")
        audio_data = handler.synthesize(texte_long.strip())
        
        if audio_data is not None and len(audio_data) > 0:
            print(f"✅ Audio généré: {len(audio_data)} échantillons")
            print(f"📊 Durée: ~{len(audio_data)/22050:.1f}s")
            
            # 6. Lecture
            print("5. 🔊 Lecture audio...")
            handler.speak(texte_long.strip())
            print("✅ Lecture terminée")
            
            print("\n🎉 SUCCÈS ! Voix française longue générée !")
            print("❓ AVEZ-VOUS ENTENDU LUXA PARLER EN FRANÇAIS NATUREL ?")
            
        else:
            print("❌ Aucun audio généré")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("🔄 Fallback vers SAPI...")
        test_sapi_francais()

def test_sapi_francais():
    """Fallback SAPI pour comparaison"""
    print("\n📢 FALLBACK SAPI (COMPARAISON)")
    print("=" * 40)
    
    try:
        sys.path.append('TTS')
        from tts_handler_mvp import TTSHandlerMVP
        
        handler = TTSHandlerMVP({'use_gpu': True})
        
        texte = "Ceci est la voix SAPI Hortense pour comparaison. La voix Piper devrait être plus naturelle."
        
        print(f"🗣️ SAPI: {texte}")
        handler.speak(texte)
        print("✅ SAPI terminé")
        
    except Exception as e:
        print(f"❌ Erreur SAPI: {e}")

if __name__ == "__main__":
    test_vraie_solution_francaise() 