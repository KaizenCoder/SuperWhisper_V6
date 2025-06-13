#!/usr/bin/env python3
"""
VALIDATION FACTUELLE - mvp_settings.yaml
Test pour vérifier que la configuration utilise RTX 3090 (CUDA:0)
"""

import yaml
import torch
import os

def test_mvp_settings_config():
    """Test factuel de la configuration mvp_settings.yaml"""
    print("🔍 VALIDATION - mvp_settings.yaml")
    print("="*40)
    
    # Test configuration
    config_path = "docs/Transmission_coordinateur/Transmission_coordinateur_20250610_1744/mvp_settings.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Fichier lu avec succès: {config_path}")
        
        # Vérifier configuration STT
        stt_device = config.get('stt', {}).get('gpu_device')
        print(f"   STT gpu_device: {stt_device}")
        
        if stt_device == "cuda:0":
            print("   ✅ STT utilise CUDA:0 (RTX 3090)")
        else:
            print(f"   ❌ STT utilise {stt_device} (INCORRECT)")
            return False
        
        # Vérifier configuration LLM
        llm_index = config.get('llm', {}).get('gpu_device_index')
        print(f"   LLM gpu_device_index: {llm_index}")
        
        if llm_index == 0:
            print("   ✅ LLM utilise index 0 (RTX 3090)")
        else:
            print(f"   ❌ LLM utilise index {llm_index} (INCORRECT)")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur lecture config: {e}")
        return False

def test_gpu_allocation():
    """Test factuel d'allocation GPU selon config"""
    print("\n🎮 TEST ALLOCATION GPU")
    print("="*30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible")
        return False
    
    try:
        # Test CUDA:0 (RTX 3090 selon config)
        device = torch.device('cuda:0')
        test_tensor = torch.randn(100, 100).to(device)
        gpu_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        
        print(f"   Device testé: cuda:0")
        print(f"   GPU utilisée: {gpu_name}")
        print(f"   Mémoire allouée: {memory_allocated:.1f}MB")
        
        is_rtx3090 = "3090" in gpu_name
        print(f"   RTX 3090 confirmée: {'✅ OUI' if is_rtx3090 else '❌ NON'}")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
        
        return is_rtx3090
        
    except Exception as e:
        print(f"❌ Erreur allocation GPU: {e}")
        return False

if __name__ == "__main__":
    print("🚨 VALIDATION mvp_settings.yaml - RTX 3090")
    print("="*50)
    
    # Test configuration
    config_valid = test_mvp_settings_config()
    
    # Test allocation GPU
    gpu_valid = test_gpu_allocation()
    
    # Résultat final
    print(f"\n🎯 RÉSULTAT FINAL:")
    print(f"   Configuration correcte: {'✅' if config_valid else '❌'}")
    print(f"   RTX 3090 utilisée: {'✅' if gpu_valid else '❌'}")
    
    overall_success = config_valid and gpu_valid
    print(f"   Validation globale: {'✅ RÉUSSIE' if overall_success else '❌ ÉCHEC'}")
    
    if overall_success:
        print("   ✅ mvp_settings.yaml utilise correctement RTX 3090")
    else:
        print("   ❌ mvp_settings.yaml nécessite correction") 