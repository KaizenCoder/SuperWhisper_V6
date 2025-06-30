#!/usr/bin/env python3
"""
Vérification finale de la configuration GPU

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

print("=== TEST SANS CONFIGURATION ===")
# Test sans rien
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    del os.environ['CUDA_VISIBLE_DEVICES']
if 'CUDA_DEVICE_ORDER' in os.environ:
    del os.environ['CUDA_DEVICE_ORDER']

# Recharger torch
import importlib
importlib.reload(torch.cuda)

print(f"Nombre de GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")

print("\n=== TEST AVEC CONFIGURATION RTX 3090 ===")
# Configuration pour RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Recharger torch pour appliquer les changements
importlib.reload(torch.cuda)

print(f"CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
print(f"CUDA_DEVICE_ORDER = '{os.environ.get('CUDA_DEVICE_ORDER')}'")
print(f"Nombre de GPU visibles: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU 0 (après configuration): {gpu_name} - {gpu_memory:.1f}GB")
    
    if "RTX 3090" in gpu_name and gpu_memory >= 20:
        print("\n✅ SUCCÈS: RTX 3090 correctement configurée!")
        print("   La configuration fonctionne avec:")
        print("   - CUDA_VISIBLE_DEVICES='1'")
        print("   - CUDA_DEVICE_ORDER='PCI_BUS_ID'")
    else:
        print("\n❌ ÉCHEC: Ce n'est pas la RTX 3090!") 