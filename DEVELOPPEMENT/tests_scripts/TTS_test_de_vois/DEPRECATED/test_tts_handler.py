#!/usr/bin/env python3
"""
Test du TTSHandler avec le modèle fr_FR-siwis-medium

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

import yaml
import sys
from pathlib import Path

# Ajouter le répertoire courant au PYTHONPATH
sys.path.append(str(Path(__file__).parent))

def test_tts_handler():
    """Test du TTSHandler avec le modèle siwis"""
    
    print("🧪 Test du TTSHandler avec modèle fr_FR-siwis-medium")
    print("=" * 60)
    
    try:
        # Charger la configuration
        with open("Config/mvp_settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration chargée")
        print(f"📍 Modèle configuré: {config['tts']['model_path']}")
        
        # Vérifier que le modèle existe
        model_path = Path(config['tts']['model_path'])
        if not model_path.exists():
            print(f"❌ ERREUR: Modèle non trouvé: {model_path}")
            return False
            
        config_path = Path(f"{config['tts']['model_path']}.json")
        if not config_path.exists():
            print(f"❌ ERREUR: Configuration du modèle non trouvée: {config_path}")
            return False
            
        print("✅ Fichiers de modèle trouvés")
        
        # Importer et initialiser le TTSHandler
        from TTS.tts_handler import TTSHandler
        
        print("\n🔧 Initialisation du TTSHandler...")
        tts_handler = TTSHandler(config['tts'])
        
        print("\n🎵 Test de synthèse vocale...")
        test_phrases = [
            "Bonjour, je suis LUXA, votre assistant vocal intelligent.",
            "Test de synthèse vocale avec le modèle français.",
            "La synthèse fonctionne parfaitement!"
        ]
        
        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n--- Test {i}/3 ---")
            tts_handler.speak(phrase)
            
            # Petite pause entre les tests
            input("Appuyez sur Entrée pour continuer...")
        
        print("\n✅ Tous les tests de synthèse ont été effectués avec succès!")
        return True
        
    except ImportError as e:
        print(f"❌ ERREUR d'import: {e}")
        print("Vérifiez que piper-tts est correctement installé.")
        return False
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        print(f"Détails: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_tts_handler()
    
    if success:
        print("\n🎉 Test terminé avec succès!")
        print("Le TTSHandler est prêt pour l'intégration dans run_assistant.py")
    else:
        print("\n❌ Test échoué!")
        print("Vérifiez l'installation de piper-tts et la configuration.")
        sys.exit(1) 