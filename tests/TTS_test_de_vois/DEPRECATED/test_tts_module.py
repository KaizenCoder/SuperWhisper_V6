#!/usr/bin/env python3
"""
Test du module TTS/ - Synthèse vocale française
🎵 Test de validation du module TTS principal

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

sys.path.append('.')

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import yaml
from TTS.tts_handler import TTSHandler

def test_tts_module():
    """Test complet du module TTS/"""
    print("🎵 TEST DU MODULE TTS/ - SYNTHÈSE VOCALE")
    print("=" * 50)
    
    try:
        # 1. Charger la configuration
        print("📋 Chargement de la configuration...")
        with open('Config/mvp_settings.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"📁 Modèle TTS: {config['tts']['model_path']}")
        
        # 2. Initialiser le handler TTS
        print("🔊 Initialisation du handler TTS...")
        tts = TTSHandler(config['tts'])
        print("✅ Handler TTS initialisé avec succès!")
        
        # 3. Test de synthèse courte
        print("\n🎤 Test 1: Phrase courte")
        test_text_1 = "Bonjour ! Je suis LUXA."
        print(f"📝 Texte: '{test_text_1}'")
        tts.speak(test_text_1)
        
        # 4. Test de synthèse longue
        print("\n🎤 Test 2: Phrase longue")
        test_text_2 = "Bonjour ! Je suis LUXA, votre assistant vocal local et confidentiel. Ce test valide le module TTS avec une voix française de qualité professionnelle."
        print(f"📝 Texte: '{test_text_2}'")
        tts.speak(test_text_2)
        
        # 5. Test technique
        print("\n🎤 Test 3: Texte technique")
        test_text_3 = "Configuration RTX 3090 validée. Pipeline STT vers LLM vers TTS opérationnel. Synthèse vocale française fonctionnelle."
        print(f"📝 Texte: '{test_text_3}'")
        tts.speak(test_text_3)
        
        print("\n✅ TOUS LES TESTS TTS RÉUSSIS!")
        print("🎵 Module TTS/ validé avec voix française")
        
    except Exception as e:
        print(f"❌ Erreur lors du test TTS: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_tts_module()
    if success:
        print("\n🎯 RÉSULTAT: Module TTS/ opérationnel!")
    else:
        print("\n❌ RÉSULTAT: Problème avec le module TTS/") 