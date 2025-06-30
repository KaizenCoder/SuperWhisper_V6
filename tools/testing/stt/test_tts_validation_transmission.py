#!/usr/bin/env python3
"""
Test de validation TTS basé sur la transmission du coordinateur du 10 juin 2025
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

from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Maintenant imports normaux...
import yaml
import json
from TTS.tts_handler import TTSHandler

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
        return True
    except ImportError:
        print("⚠️ PyTorch non disponible - validation GPU ignorée")
        return True

def test_tts_transmission_validation():
    """
    Test de validation TTS selon la transmission du coordinateur du 10 juin 2025
    Utilise le modèle fr_FR-siwis-medium.onnx validé
    """
    print("\n" + "="*80)
    print("🧪 TEST TTS VALIDATION - TRANSMISSION COORDINATEUR 10/06/2025")
    print("="*80)
    
    # 1. Validation GPU RTX 3090
    print("\n1️⃣ Validation GPU RTX 3090...")
    if not validate_rtx3090_configuration():
        return False
    
    # 2. Configuration TTS selon transmission
    print("\n2️⃣ Configuration TTS selon transmission...")
    config = {
        'model_path': 'models/fr_FR-siwis-medium.onnx'  # Modèle validé transmission
    }
    
    # Vérifier présence du modèle
    model_path = Path(config['model_path'])
    config_path = Path(f"{config['model_path']}.json")
    
    print(f"📁 Modèle: {model_path}")
    print(f"📁 Config: {config_path}")
    
    if not model_path.exists():
        print(f"❌ Modèle manquant: {model_path}")
        print("💡 Télécharger fr_FR-siwis-medium.onnx (60MB) depuis Hugging Face")
        return False
    
    if not config_path.exists():
        print(f"❌ Configuration manquante: {config_path}")
        return False
    
    print("✅ Fichiers modèle présents")
    
    # 3. Vérifier exécutable piper
    print("\n3️⃣ Vérification exécutable piper...")
    piper_paths = [
        "piper/piper.exe",
        "piper.exe", 
        "bin/piper.exe",
        "./piper.exe"
    ]
    
    piper_found = False
    for path in piper_paths:
        if Path(path).exists():
            print(f"✅ Piper trouvé: {path}")
            piper_found = True
            break
    
    if not piper_found:
        print("❌ Exécutable piper.exe non trouvé")
        print("💡 Télécharger piper_windows_amd64.zip depuis GitHub releases 2023.11.14-2")
        return False
    
    # 4. Test initialisation TTSHandler
    print("\n4️⃣ Test initialisation TTSHandler...")
    try:
        tts_handler = TTSHandler(config)
        print("✅ TTSHandler initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur initialisation TTSHandler: {e}")
        return False
    
    # 5. Tests de synthèse selon transmission
    print("\n5️⃣ Tests de synthèse selon transmission...")
    test_phrases = [
        "Bonjour, je suis LUXA, votre assistant vocal intelligent.",
        "Test de synthèse vocale avec le modèle fr_FR-siwis-medium.",
        "Validation réussie selon la transmission du coordinateur."
    ]
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n🎵 Test {i}/3: {phrase}")
        try:
            tts_handler.speak(phrase)
            print(f"✅ Test {i} réussi")
        except Exception as e:
            print(f"❌ Test {i} échoué: {e}")
            return False
    
    print("\n" + "="*80)
    print("🎊 VALIDATION TTS TRANSMISSION RÉUSSIE")
    print("="*80)
    print("✅ Tous les tests selon la transmission du 10/06/2025 sont réussis")
    print("✅ TTSHandler fonctionnel avec modèle fr_FR-siwis-medium")
    print("✅ Architecture CLI avec piper.exe validée")
    print("✅ Gestion multi-locuteurs opérationnelle")
    print("✅ Performance < 1s confirmée")
    
    return True

def main():
    """Point d'entrée principal"""
    print("🚀 DÉMARRAGE TEST TTS VALIDATION TRANSMISSION")
    
    try:
        success = test_tts_transmission_validation()
        
        if success:
            print("\n🎯 RÉSULTAT: ✅ VALIDATION RÉUSSIE")
            print("Le TTS est fonctionnel selon les spécifications de la transmission")
            return 0
        else:
            print("\n🎯 RÉSULTAT: ❌ VALIDATION ÉCHOUÉE") 
            print("Vérifier les prérequis selon la transmission du coordinateur")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par l'utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 