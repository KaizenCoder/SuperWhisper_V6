#!/usr/bin/env python3
"""
🏆 VALIDATION COMPLÈTE RTX 3090 - Script de Test
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Script de validation pour vérifier la configuration GPU RTX 3090 dans SuperWhisper V6
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 Ti (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Maintenant imports normaux...
import torch
import subprocess
import traceback
from pathlib import Path

def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

def test_script_gpu_configuration(script_path: str) -> tuple[bool, str]:
    """Teste si un script Python a la configuration GPU RTX 3090 correcte"""
    try:
        if not Path(script_path).exists():
            return False, f"Fichier introuvable: {script_path}"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifications critiques (patterns flexibles)
        checks = [
            ("CUDA_VISIBLE_DEVICES", "1", "Configuration RTX 3090"),
            ("CUDA_DEVICE_ORDER", "PCI_BUS_ID", "Ordre GPU stable"),
            ("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb", "Optimisation mémoire"),
            ("def validate_rtx3090", "", "Fonction validation"),
            ("#!/usr/bin/env python3", "", "Shebang standard")
        ]
        
        missing = []
        for pattern1, pattern2, desc in checks:
            if pattern2:  # Deux patterns à vérifier
                if pattern1 not in content or pattern2 not in content:
                    missing.append(f"{desc} ({pattern1}={pattern2})")
            else:  # Un seul pattern
                if pattern1 not in content:
                    missing.append(f"{desc} ({pattern1})")
        
        # Vérifier qu'il n'y a pas d'anciennes configurations incorrectes
        bad_patterns = [
            ("CUDA_VISIBLE_DEVICES='0'", "RTX 5060 Ti interdite!"),
            ("device='cuda:1'", "Device cuda:1 direct incorrect"),
        ]
        
        warnings = []
        for pattern, desc in bad_patterns:
            if pattern in content:
                warnings.append(f"⚠️ {desc}: {pattern}")
        
        if missing:
            return False, f"Configuration incomplète: {', '.join(missing)}"
        
        if warnings:
            return False, f"Configuration dangereuse: {', '.join(warnings)}"
        
        return True, "Configuration GPU RTX 3090 correcte"
        
    except Exception as e:
        return False, f"Erreur lecture: {str(e)}"

def test_script_execution(script_path: str) -> tuple[bool, str]:
    """Teste l'exécution d'un script avec validation RTX 3090"""
    try:
        # Préparer environnement propre avec configuration RTX 3090
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '1'
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        if result.returncode == 0:
            output = result.stdout
            # Vérifier que RTX 3090 est bien détectée
            if "RTX 3090" in output and ("✅" in output or "validée" in output):
                return True, "Script exécuté avec succès - RTX 3090 détectée"
            else:
                return False, f"RTX 3090 non détectée dans l'output: {output[:200]}"
        else:
            return False, f"Échec exécution (code {result.returncode}): {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout exécution (>30s)"
    except Exception as e:
        return False, f"Erreur exécution: {str(e)}"

def validate_superwhisper_scripts():
    """Valide plusieurs scripts clés de SuperWhisper V6"""
    print("🔍 VALIDATION SCRIPTS SUPERWHISPER V6")
    print("=" * 50)
    
    # Scripts prioritaires à tester
    scripts_to_test = [
        "test_rtx3090_detection.py",
        "test_cuda_debug.py", 
        "test_cuda.py",
        "test_gpu_correct.py",
        "test_gpu_verification.py",
        "test_rtx3090_access.py"
    ]
    
    results = {}
    
    for script in scripts_to_test:
        print(f"\n🔍 Test: {script}")
        
        # Test 1: Configuration
        config_ok, config_msg = test_script_gpu_configuration(script)
        print(f"   📋 Config: {'✅' if config_ok else '❌'} {config_msg}")
        
        # Test 2: Exécution (seulement si config OK)
        if config_ok:
            exec_ok, exec_msg = test_script_execution(script)
            print(f"   🚀 Exécution: {'✅' if exec_ok else '❌'} {exec_msg}")
        else:
            exec_ok = False
            print(f"   🚀 Exécution: ⏸️ Ignorée (config incorrecte)")
        
        results[script] = {
            'config_ok': config_ok,
            'exec_ok': exec_ok,
            'functional': config_ok and exec_ok
        }
        
        status = "✅ FONCTIONNEL" if results[script]['functional'] else "❌ PROBLÈME"
        print(f"   🏆 Statut: {status}")
    
    return results

def test_gpu_system_validation():
    """Validation complète du système GPU"""
    print("\n🎮 VALIDATION SYSTÈME GPU RTX 3090")
    print("=" * 45)
    
    try:
        validate_rtx3090_mandatory()
        
        # Tests supplémentaires
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"   🔢 Devices visibles: {device_count}")
        print(f"   🎯 Device actuel: {current_device}")
        print(f"   🎮 GPU: {gpu_name}")
        print(f"   💾 VRAM: {gpu_memory:.1f}GB")
        
        # Test allocation mémoire
        test_tensor = torch.randn(1000, 1000, device='cuda')
        print(f"   ✅ Test allocation CUDA réussi")
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur système GPU: {e}")
        return False

def main():
    """Fonction principale de validation RTX 3090"""
    print("🏆 VALIDATION COMPLÈTE RTX 3090 - SUPERWHISPER V6")
    print("=" * 60)
    print(f"📅 Test en cours...")
    print()
    
    try:
        # Test 1: Système GPU
        gpu_system_ok = test_gpu_system_validation()
        
        # Test 2: Scripts SuperWhisper
        scripts_results = validate_superwhisper_scripts()
        
        # Analyse résultats
        total_scripts = len(scripts_results)
        functional_scripts = sum(1 for r in scripts_results.values() if r['functional'])
        
        print(f"\n📊 RÉSUMÉ VALIDATION")
        print("=" * 30)
        print(f"🎮 Système GPU: {'✅ OK' if gpu_system_ok else '❌ ÉCHEC'}")
        print(f"📝 Scripts testés: {total_scripts}")
        print(f"✅ Scripts fonctionnels: {functional_scripts}")
        print(f"❌ Scripts problématiques: {total_scripts - functional_scripts}")
        print(f"📈 Taux de réussite: {functional_scripts/total_scripts*100:.1f}%")
        
        # Détail des problèmes
        problematic = [name for name, result in scripts_results.items() if not result['functional']]
        if problematic:
            print(f"\n⚠️ SCRIPTS À CORRIGER:")
            for script in problematic:
                result = scripts_results[script]
                config_status = "✅" if result['config_ok'] else "❌"
                exec_status = "✅" if result['exec_ok'] else "❌"
                print(f"   📄 {script} - Config:{config_status} Exec:{exec_status}")
        
        # Conclusion
        overall_success = gpu_system_ok and functional_scripts == total_scripts
        print(f"\n🎯 VALIDATION GLOBALE: {'✅ RÉUSSIE' if overall_success else '⚠️ PARTIELLE'}")
        
        if overall_success:
            print("🎉 Configuration RTX 3090 parfaitement opérationnelle!")
        else:
            print("🔧 Corrections nécessaires pour finaliser la mission GPU")
        
        return overall_success
        
    except Exception as e:
        print(f"\n🚨 ERREUR CRITIQUE VALIDATION: {e}")
        traceback.print_exc()
        return False

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory()
    success = main()
    sys.exit(0 if success else 1) 