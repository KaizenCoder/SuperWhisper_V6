#!/usr/bin/env python3
"""
Script de validation centralisé pour la configuration GPU
🚨 VALIDATION CRITIQUE: RTX 3090 (CUDA:0) OBLIGATOIRE

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
from pathlib import Path
from typing import List, Dict, Tuple
import importlib.util
import ast

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:0) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Liste des fichiers à valider
FILES_TO_VALIDATE = [
    # Modules Core
    "benchmarks/benchmark_stt_realistic.py",
    "LLM/llm_manager_enhanced.py",
    "LUXA_TTS/tts_handler_coqui.py",
    "Orchestrator/fallback_manager.py",
    "STT/vad_manager_optimized.py",
    "TTS/tts_handler_coqui.py",
    "TTS/tts_handler_piper_native.py",
    # Scripts de Test
    "tests/test_double_check_corrections.py",
    "tests/test_double_check_validation_simple.py",
    "test_cuda_debug.py",
    "test_cuda.py",
    "test_espeak_french.py",
    "test_french_voice.py",
    "test_gpu_correct.py",
    "test_piper_native.py",
    "test_tts_fixed.py",
    "test_tts_long_feedback.py",
    "test_upmc_model.py",
    "test_validation_decouverte.py",
    "TTS/tts_handler_piper_rtx3090.py"
]

def validate_rtx3090_system():
    """Validation système RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in gpu_name:
        raise RuntimeError(f"🚫 GPU détecté: {gpu_name} - RTX 3090 requise")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) insuffisante - RTX 3090 requise")
    
    print(f"✅ RTX 3090 système validée: {gpu_name} ({gpu_memory:.1f}GB)")

def check_file_gpu_config(filepath: str) -> Dict[str, any]:
    """Vérifie la configuration GPU dans un fichier"""
    results = {
        'file': filepath,
        'exists': False,
        'cuda_visible_devices': None,
        'cuda_references': [],
        'errors': [],
        'warnings': []
    }
    
    file_path = Path(filepath)
    if not file_path.exists():
        results['errors'].append(f"Fichier non trouvé: {filepath}")
        return results
    
    results['exists'] = True
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Recherche CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in content:
            import re
            matches = re.findall(r"CUDA_VISIBLE_DEVICES['\"]?\s*[=:]\s*['\"]?(\d+)", content)
            if matches:
                results['cuda_visible_devices'] = matches[0]
                if matches[0] != '1':
                    results['errors'].append(f"❌ CUDA_VISIBLE_DEVICES='{matches[0]}' (doit être '1')")
                else:
                    results['warnings'].append(f"✅ CUDA_VISIBLE_DEVICES='1' correct")
        
        # Recherche références cuda:X
        cuda_refs = re.findall(r'cuda:(\d+)', content)
        if cuda_refs:
            results['cuda_references'] = cuda_refs
            for ref in cuda_refs:
                if ref != '0':
                    results['errors'].append(f"❌ Référence 'cuda:{ref}' trouvée (doit être 'cuda:0')")
        
        # Recherche torch.cuda.set_device
        device_sets = re.findall(r'torch\.cuda\.set_device\((\d+)\)', content)
        for device in device_sets:
            if device != '0':
                results['errors'].append(f"❌ torch.cuda.set_device({device}) trouvé (doit être 0)")
                
    except Exception as e:
        results['errors'].append(f"Erreur lecture fichier: {e}")
    
    return results

def generate_report(results: List[Dict]) -> None:
    """Génère un rapport de validation"""
    print("\n" + "="*80)
    print("📊 RAPPORT DE VALIDATION GPU - SUPERWHISPER V6")
    print("="*80)
    
    total_files = len(results)
    files_with_errors = sum(1 for r in results if r['errors'])
    files_ok = total_files - files_with_errors
    
    print(f"\n📈 RÉSUMÉ:")
    print(f"  - Fichiers analysés: {total_files}")
    print(f"  - ✅ Fichiers OK: {files_ok}")
    print(f"  - ❌ Fichiers avec erreurs: {files_with_errors}")
    
    if files_with_errors > 0:
        print(f"\n🚨 FICHIERS À CORRIGER ({files_with_errors}):")
        for result in results:
            if result['errors']:
                print(f"\n  📁 {result['file']}:")
                for error in result['errors']:
                    print(f"    {error}")
    
    print("\n" + "="*80)
    
    # Sauvegarder le rapport
    report_path = Path("docs/gpu-correction/validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE VALIDATION GPU - SUPERWHISPER V6\n")
        f.write("="*80 + "\n\n")
        for result in results:
            f.write(f"Fichier: {result['file']}\n")
            f.write(f"Statut: {'❌ ERREURS' if result['errors'] else '✅ OK'}\n")
            if result['errors']:
                for error in result['errors']:
                    f.write(f"  - {error}\n")
            f.write("\n")
    
    print(f"📄 Rapport sauvegardé: {report_path}")

def main():
    """Fonction principale de validation"""
    print("🔍 VALIDATION GPU SUPERWHISPER V6")
    print("="*50)
    
    # Validation système
    try:
        validate_rtx3090_system()
    except Exception as e:
        print(f"❌ ERREUR SYSTÈME: {e}")
        sys.exit(1)
    
    # Validation des fichiers
    results = []
    for filepath in FILES_TO_VALIDATE:
        print(f"\n🔍 Analyse: {filepath}")
        result = check_file_gpu_config(filepath)
        results.append(result)
        
        if result['errors']:
            print(f"  ❌ {len(result['errors'])} erreur(s) trouvée(s)")
        else:
            print(f"  ✅ Configuration OK")
    
    # Génération du rapport
    generate_report(results)
    
    # Code de sortie
    if any(r['errors'] for r in results):
        print("\n❌ VALIDATION ÉCHOUÉE - Corrections requises")
        sys.exit(1)
    else:
        print("\n✅ VALIDATION RÉUSSIE - Tous les fichiers sont corrects")
        sys.exit(0)

if __name__ == "__main__":
    main() 