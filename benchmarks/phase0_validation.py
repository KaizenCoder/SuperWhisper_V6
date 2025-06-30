#!/usr/bin/env python3
"""
Phase 0 Validation - Luxa SuperWhisper_V6
==========================================

Ce script valide que la configuration initiale du projet Luxa est correcte.

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

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_status(message, status="INFO"):
    symbols = {"INFO": "ℹ️", "OK": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def check_directory_structure():
    print_header("VALIDATION STRUCTURE DU PROJET")
    
    required_dirs = [
        "STT", "LLM", "TTS", "Orchestrator", 
        "Config", "Tests", "Logs", "benchmarks"
    ]
    
    required_files = ["run_assistant.py"]
    
    all_good = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print_status(f'Dossier "{directory}" trouvé', "OK")
        else:
            print_status(f'Dossier "{directory}" manquant', "ERROR")
            all_good = False
    
    for file in required_files:
        if Path(file).exists():
            print_status(f'Fichier "{file}" trouvé', "OK")
        else:
            print_status(f'Fichier "{file}" manquant', "ERROR")
            all_good = False
    
    return all_good

def main():
    print_header("LUXA SUPERWHISPER_V6 - PHASE 0 VALIDATION")
    print_status("Démarrage de la validation de configuration...")
    
    structure_ok = check_directory_structure()
    
    print_header("RÉSULTAT DE LA VALIDATION")
    
    if structure_ok:
        print_status("🎉 VALIDATION RÉUSSIE - Configuration Phase 0 OK!", "OK")
        print_status("Vous pouvez passer au développement des modules.", "INFO")
        return 0
    else:
        print_status("❌ VALIDATION ÉCHOUÉE - Problèmes détectés", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 