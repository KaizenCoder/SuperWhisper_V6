#!/usr/bin/env python3
"""
VALIDATION CORRECTION 3 : tests/test_llm_handler.py
🚨 CONFIGURATION GPU: RTX 3090 (INDEX 0) OBLIGATOIRE

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

import torch

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # RTX 3090 EXCLUSIVEMENT
print("🎮 Test Validation: RTX 3090 (CUDA:0) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def validate_llm_handler_correction():
    """Validation que test_llm_handler utilise RTX 3090 (index 0)"""
    print("🔍 VALIDATION CORRECTION - tests/test_llm_handler.py")
    
    # 1. Vérifier le contenu du fichier
    with open('tests/test_llm_handler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 2. Vérifications critiques
    violations = []
    
    # Vérifier que gpu_device_index = 0
    if "'gpu_device_index': 0" not in content:
        violations.append("❌ gpu_device_index: 0 non trouvé - RTX 3090 non configurée")
    
    # Vérifier qu'il n'y a plus d'index 1 non documenté
    if "'gpu_device_index': 1" in content and 'RTX 5060' not in content:
        violations.append("❌ gpu_device_index: 1 trouvé sans documentation RTX 5060")
    
    # Vérifier les commentaires corrects
    if 'RTX 3090 (CUDA:0)' not in content:
        violations.append("❌ Documentation RTX 3090 (CUDA:0) manquante")
    
    # 3. Validation GPU si disponible
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU détectée: {gpu_name}")
        print(f"✅ VRAM disponible: {gpu_memory:.1f}GB")
        
        # VALIDATION CRITIQUE
        if "RTX 3090" not in gpu_name:
            violations.append(f"❌ GPU incorrecte détectée: {gpu_name}")
        if gpu_memory < 20:
            violations.append(f"❌ VRAM insuffisante: {gpu_memory}GB < 20GB")
    else:
        print("⚠️ CUDA non disponible - validation GPU impossible")
    
    # 4. Rapport
    if violations:
        print("🚫 ÉCHEC VALIDATION:")
        for violation in violations:
            print(f"   {violation}")
        return False
    else:
        print("✅ VALIDATION RÉUSSIE:")
        print("   ✅ tests/test_llm_handler.py utilise gpu_device_index: 0")
        print("   ✅ Documentation RTX 3090 (CUDA:0) présente")
        if torch.cuda.is_available():
            print("   ✅ RTX 3090 confirmée matériellement")
        return True

if __name__ == "__main__":
    success = validate_llm_handler_correction()
    if success:
        print("\n🎉 CORRECTION 3 VALIDÉE AVEC SUCCÈS")
        sys.exit(0)
    else:
        print("\n🚫 CORRECTION 3 ÉCHOUÉE")
        sys.exit(1) 