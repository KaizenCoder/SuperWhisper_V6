#!/usr/bin/env python3
"""
Analyse de la configuration GPU existante dans les 38 fichiers
Mission : Homogénéisation GPU SuperWhisper V6

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

import re
from pathlib import Path
import json

print("🔍 ANALYSE CONFIGURATION GPU - 38 fichiers")
print("=" * 50)

# Liste des fichiers analysés (38 fichiers sauvegardés avec succès)
files_to_analyze = [
    "benchmarks/benchmark_stt_realistic.py",
    "LLM/llm_manager_enhanced.py", 
    "LUXA_TTS/tts_handler_coqui.py",
    "Orchestrator/fallback_manager.py",
    "STT/vad_manager_optimized.py",
    "STT/stt_manager_robust.py",
    "STT/vad_manager.py",
    "TTS/tts_handler_piper_espeak.py",
    "TTS/tts_handler_piper_fixed.py",
    "TTS/tts_handler_piper_french.py",
    "utils/gpu_manager.py",
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
    "TTS/tts_handler_piper_rtx3090.py",
    "tests/test_llm_handler.py",
    "tests/test_stt_handler.py",
    "test_correction_validation_1.py",
    "test_correction_validation_2.py",
    "test_correction_validation_3.py",
    "test_correction_validation_4.py",
    "test_rtx3090_detection.py",
    "test_tts_rtx3090_performance.py",
    "test_validation_globale_finale.py",
    "test_validation_mvp_settings.py",
    "test_validation_rtx3090_detection.py",
    "test_validation_stt_manager_robust.py",
    "test_validation_tts_performance.py",
    "validate_gpu_config.py"
]

# Patterns à chercher
patterns = {
    'cuda_visible_devices': re.compile(r"os\.environ\[['\"]CUDA_VISIBLE_DEVICES['\"]\]\s*=\s*['\"]([^'\"]+)['\"]"),
    'cuda_device_order': re.compile(r"os\.environ\[['\"]CUDA_DEVICE_ORDER['\"]\]\s*=\s*['\"]([^'\"]+)['\"]"),
    'cuda_usage': re.compile(r"['\"]cuda(?::(\d+))?['\"]"),
    'device_cuda': re.compile(r"device\s*=\s*['\"]cuda(?::(\d+))?['\"]"),
    'torch_device': re.compile(r"torch\.device\(['\"]cuda(?::(\d+))?['\"]\)"),
    'set_device': re.compile(r"torch\.cuda\.set_device\((\d+)\)"),
    'device_map': re.compile(r"device_map\s*=\s*\{['\"]?['\"]?\s*:\s*(\d+)\}"),
    'gpu_device_index': re.compile(r"gpu_device_index\s*[:=]\s*(\d+)")
}

def analyze_file(file_path):
    """Analyser un fichier pour la configuration GPU"""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Erreur lecture {file_path}: {e}")
        return None
    
    analysis = {
        'file': file_path,
        'has_cuda_visible_devices': False,
        'cuda_visible_devices_value': None,
        'has_cuda_device_order': False,
        'cuda_device_order_value': None,
        'cuda_usages': [],
        'device_configurations': [],
        'potential_issues': []
    }
    
    # Analyser CUDA_VISIBLE_DEVICES
    match = patterns['cuda_visible_devices'].search(content)
    if match:
        analysis['has_cuda_visible_devices'] = True
        analysis['cuda_visible_devices_value'] = match.group(1)
    
    # Analyser CUDA_DEVICE_ORDER
    match = patterns['cuda_device_order'].search(content)
    if match:
        analysis['has_cuda_device_order'] = True
        analysis['cuda_device_order_value'] = match.group(1)
    
    # Analyser utilisation CUDA
    for pattern_name, pattern in patterns.items():
        if pattern_name in ['cuda_visible_devices', 'cuda_device_order']:
            continue
            
        matches = pattern.findall(content)
        if matches:
            if pattern_name == 'cuda_usage':
                analysis['cuda_usages'].extend(matches)
            else:
                analysis['device_configurations'].extend([(pattern_name, match) for match in matches])
    
    # Détecter problèmes potentiels
    if not analysis['has_cuda_visible_devices']:
        analysis['potential_issues'].append("CUDA_VISIBLE_DEVICES manquant")
    elif analysis['cuda_visible_devices_value'] != '1':
        analysis['potential_issues'].append(f"CUDA_VISIBLE_DEVICES='{analysis['cuda_visible_devices_value']}' (attendu: '1')")
    
    if not analysis['has_cuda_device_order']:
        analysis['potential_issues'].append("CUDA_DEVICE_ORDER manquant")
    elif analysis['cuda_device_order_value'] != 'PCI_BUS_ID':
        analysis['potential_issues'].append(f"CUDA_DEVICE_ORDER='{analysis['cuda_device_order_value']}' (attendu: 'PCI_BUS_ID')")
    
    # Vérifier utilisation cohérente de cuda:0
    for device_config in analysis['device_configurations']:
        pattern_name, value = device_config
        if value and value != '0':
            analysis['potential_issues'].append(f"Utilisation {pattern_name} avec device {value} (attendu: 0 après mapping)")
    
    return analysis

def main():
    results = []
    files_found = 0
    files_with_issues = 0
    
    print(f"📊 Analyse de {len(files_to_analyze)} fichiers...\n")
    
    for file_path in files_to_analyze:
        print(f"🔍 Analyse : {file_path}")
        
        analysis = analyze_file(file_path)
        if analysis is None:
            print(f"   ⚠️ Fichier non trouvé ou erreur")
            continue
            
        files_found += 1
        results.append(analysis)
        
        # Afficher résumé
        status_icon = "✅" if not analysis['potential_issues'] else "⚠️"
        print(f"   {status_icon} CUDA_VISIBLE_DEVICES: {analysis['cuda_visible_devices_value'] or 'NON'}")
        print(f"   {status_icon} CUDA_DEVICE_ORDER: {analysis['cuda_device_order_value'] or 'NON'}")
        
        if analysis['potential_issues']:
            files_with_issues += 1
            print(f"   🚨 Problèmes: {len(analysis['potential_issues'])}")
            for issue in analysis['potential_issues']:
                print(f"      - {issue}")
        
        print()
    
    # Sauvegarde rapport détaillé
    report_path = "docs/gpu-correction/reports/gpu_config_analysis.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Générer résumé
    print("=" * 50)
    print("📈 RÉSUMÉ ANALYSE")
    print("=" * 50)
    print(f"📊 Fichiers analysés: {files_found}/{len(files_to_analyze)}")
    print(f"✅ Fichiers sans problème: {files_found - files_with_issues}")
    print(f"⚠️ Fichiers avec problèmes: {files_with_issues}")
    
    # Statistiques par problème
    issue_stats = {}
    for result in results:
        for issue in result['potential_issues']:
            issue_stats[issue] = issue_stats.get(issue, 0) + 1
    
    if issue_stats:
        print("\n🔍 PROBLÈMES DÉTECTÉS:")
        for issue, count in sorted(issue_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {count:2d}x {issue}")
    
    print(f"\n📄 Rapport détaillé: {report_path}")
    
    # Recommandations
    print("\n🎯 RECOMMANDATIONS:")
    if files_with_issues > 0:
        print("  1. Ajouter CUDA_VISIBLE_DEVICES='1' dans tous les fichiers")
        print("  2. Ajouter CUDA_DEVICE_ORDER='PCI_BUS_ID' dans tous les fichiers")
        print("  3. S'assurer que le code utilise cuda:0 après cette configuration")
        print("  4. Ajouter fonction validate_rtx3090_mandatory() dans chaque fichier")
    else:
        print("  ✅ Tous les fichiers ont une configuration GPU correcte !")

if __name__ == "__main__":
    main() 