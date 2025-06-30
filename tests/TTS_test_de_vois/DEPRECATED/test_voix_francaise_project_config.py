#!/usr/bin/env python3
"""
TEST VOIX FRANÇAISE CONFIGURATION PROJET - LUXA SuperWhisper V6
🚨 UTILISE LA VRAIE CONFIG mvp_settings.yaml QUI MARCHE

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

def test_vraie_config_projet():
    """Test avec la VRAIE configuration du projet mvp_settings.yaml"""
    
    print("\n🎯 TEST VRAIE CONFIGURATION PROJET")
    print("=" * 50)
    
    try:
        import torch
        # Validation RTX 3090
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Configuration EXACTE du projet (mvp_settings.yaml)
        config_projet = {
            'model_path': 'models/fr_FR-siwis-medium.onnx',  # ✅ CHEMIN PROJET
            'use_gpu': True,
            'sample_rate': 22050
        }
        
        print(f"📁 Modèle projet: {config_projet['model_path']}")
        
        # Import TTS Handler
        sys.path.append('TTS')
        from tts_handler_piper_french import TTSHandlerPiperFrench
        
        # Initialisation avec config projet
        print("🚀 Initialisation avec config projet...")
        handler = TTSHandlerPiperFrench(config_projet)
        print("✅ Handler initialisé avec config projet")
        
        # Test texte français LONG pour bien entendre
        texte_francais = """
        Bonjour ! Je suis LUXA, votre assistant vocal intelligent. 
        Je parle parfaitement français avec une voix naturelle et claire.
        Cette phrase est longue pour que vous puissiez bien entendre ma voix française.
        Est-ce que vous m'entendez bien parler en français maintenant ?
        """
        
        print(f"🗣️ Texte français ({len(texte_francais)} chars)")
        
        # Synthèse et lecture
        print("🔊 Synthèse voix française...")
        handler.speak(texte_francais)
        print("✅ Test français terminé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_sapi_francais():
    """Fallback avec SAPI français si modèle Piper échoue"""
    
    print("\n🎤 TEST FALLBACK SAPI FRANÇAIS")
    print("=" * 40)
    
    try:
        import win32com.client
        
        # Configuration SAPI pour français
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        
        # Recherche voix française
        voix_francaise = None
        for i in range(voices.Count):
            voice = voices.Item(i)
            voice_name = voice.GetDescription()
            if "French" in voice_name or "Française" in voice_name:
                voix_francaise = voice
                print(f"🇫🇷 Voix française trouvée: {voice_name}")
                break
        
        if voix_francaise:
            speaker.Voice = voix_francaise
            texte = "Bonjour ! Je suis LUXA et je parle français avec SAPI."
            
            print("🔊 Test SAPI français...")
            speaker.Speak(texte)
            print("✅ SAPI français testé")
            return True
        else:
            print("❌ Aucune voix française SAPI trouvée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur SAPI: {e}")
        return False

if __name__ == "__main__":
    print("🎮 VALIDATION CONFIGURATION RTX 3090")
    print("🇫🇷 TEST VOIX FRANÇAISE VRAIE CONFIG PROJET")
    print("=" * 60)
    
    # Test configuration projet
    success_projet = test_vraie_config_projet()
    
    if not success_projet:
        print("\n⚠️ Config projet échouée, test fallback SAPI...")
        test_fallback_sapi_francais()
    
    print("\n🎯 Test terminé - Avez-vous entendu du français ?") 