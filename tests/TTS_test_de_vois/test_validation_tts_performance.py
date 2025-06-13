#!/usr/bin/env python3
"""
VALIDATION FACTUELLE - test_tts_rtx3090_performance.py
Test pour vérifier que le script utilise RTX 3090 (CUDA:0)
"""

import os
import torch
import subprocess
import sys

def test_tts_performance_config():
    """Test factuel de la configuration dans le script de performance"""
    print("🔍 VALIDATION - test_tts_rtx3090_performance.py")
    print("="*50)
    
    # Lire le contenu du fichier
    script_path = "test_tts_rtx3090_performance.py"
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ Fichier lu: {script_path}")
        
        # Vérifier CUDA_VISIBLE_DEVICES
        cuda_visible_line = None
        for i, line in enumerate(content.split('\n')):
            if 'CUDA_VISIBLE_DEVICES' in line and '=' in line:
                cuda_visible_line = line.strip()
                break
        
        if cuda_visible_line:
            print(f"   Configuration trouvée: {cuda_visible_line}")
            if "'0'" in cuda_visible_line:
                print("   ✅ CUDA_VISIBLE_DEVICES utilise '0' (RTX 3090)")
                config_ok = True
            else:
                print("   ❌ CUDA_VISIBLE_DEVICES n'utilise pas '0'")
                config_ok = False
        else:
            print("   ❌ CUDA_VISIBLE_DEVICES non trouvé")
            config_ok = False
        
        # Vérifier les références get_device_name
        device_refs = []
        for line in content.split('\n'):
            if 'get_device_name(' in line:
                device_refs.append(line.strip())
        
        print(f"   Références device trouvées: {len(device_refs)}")
        device_ok = True
        for ref in device_refs:
            print(f"   - {ref}")
            if 'get_device_name(0)' not in ref:
                print("     ❌ Ne référence pas device 0")
                device_ok = False
            else:
                print("     ✅ Référence device 0 (RTX 3090)")
        
        return config_ok and device_ok
        
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {e}")
        return False

def test_gpu_environment():
    """Test de l'environnement GPU"""
    print("\n🎮 TEST ENVIRONNEMENT GPU")
    print("="*30)
    
    # Clear any existing CUDA_VISIBLE_DEVICES
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible")
        return False
    
    try:
        # Test device 0 (RTX 3090)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"   Device 0: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f}GB")
        
        is_rtx3090 = "3090" in gpu_name
        print(f"   RTX 3090 confirmée: {'✅ OUI' if is_rtx3090 else '❌ NON'}")
        
        return is_rtx3090
        
    except Exception as e:
        print(f"❌ Erreur GPU: {e}")
        return False

def test_script_syntax():
    """Test de syntaxe du script"""
    print("\n🔍 TEST SYNTAXE SCRIPT")
    print("="*30)
    
    try:
        # Test compilation du script
        script_path = "test_tts_rtx3090_performance.py"
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, script_path, 'exec')
        print("   ✅ Syntaxe Python valide")
        return True
        
    except SyntaxError as e:
        print(f"   ❌ Erreur syntaxe: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Erreur compilation: {e}")
        return False

if __name__ == "__main__":
    print("🚨 VALIDATION TTS PERFORMANCE - RTX 3090")
    print("="*60)
    
    # Tests
    config_valid = test_tts_performance_config()
    gpu_valid = test_gpu_environment()
    syntax_valid = test_script_syntax()
    
    # Résultat final
    print(f"\n🎯 RÉSULTAT FINAL:")
    print(f"   Configuration script: {'✅' if config_valid else '❌'}")
    print(f"   GPU environnement: {'✅' if gpu_valid else '❌'}")
    print(f"   Syntaxe valide: {'✅' if syntax_valid else '❌'}")
    
    overall_success = config_valid and gpu_valid and syntax_valid
    print(f"   Validation globale: {'✅ RÉUSSIE' if overall_success else '❌ ÉCHEC'}")
    
    if overall_success:
        print("   ✅ test_tts_rtx3090_performance.py utilise correctement RTX 3090")
    else:
        print("   ❌ test_tts_rtx3090_performance.py nécessite correction") 