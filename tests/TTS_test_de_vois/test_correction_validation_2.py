#!/usr/bin/env python3
"""
VALIDATION CORRECTION 2 : utils/gpu_manager.py
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:0) OBLIGATOIRE
"""

import os
import sys
import torch

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # RTX 3090 EXCLUSIVEMENT
print("🎮 Test Validation: RTX 3090 (CUDA:0) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def validate_gpu_manager_correction():
    """Validation que gpu_manager utilise RTX 3090 (cuda:0)"""
    print("🔍 VALIDATION CORRECTION - utils/gpu_manager.py")
    
    # 1. Vérifier le contenu du fichier
    with open('utils/gpu_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 2. Vérifications critiques
    violations = []
    
    # Vérifier qu'il n'y a plus de cuda:1 non documenté
    if 'cuda:1' in content and 'RTX 5060' not in content:
        violations.append("❌ cuda:1 trouvé sans documentation RTX 5060")
    
    # Vérifier que cuda:0 est utilisé
    if 'cuda:0' not in content:
        violations.append("❌ cuda:0 non trouvé - RTX 3090 non configurée")
    
    # Vérifier les commentaires corrects
    if 'RTX 3090 (CUDA:0)' not in content:
        violations.append("❌ Documentation RTX 3090 (CUDA:0) manquante")
    
    # 3. Test fonctionnel du GPU Manager
    try:
        from utils.gpu_manager import GPUManager
        gpu_manager = GPUManager()
        
        # Test des méthodes get_device
        llm_device = gpu_manager.get_device("llm")
        stt_device = gpu_manager.get_device("stt")
        fallback_device = gpu_manager.get_device("fallback")
        
        print(f"📊 GPU Manager - LLM device: {llm_device}")
        print(f"📊 GPU Manager - STT device: {stt_device}")
        print(f"📊 GPU Manager - Fallback device: {fallback_device}")
        
        # Validation que tous pointent vers cuda:0
        if llm_device != "cuda:0" and llm_device != "cpu":
            violations.append(f"❌ LLM device incorrect: {llm_device} (attendu: cuda:0)")
            
        if stt_device != "cuda:0" and stt_device != "cpu":
            violations.append(f"❌ STT device incorrect: {stt_device} (attendu: cuda:0)")
            
        if fallback_device != "cuda:0" and fallback_device != "cpu":
            violations.append(f"❌ Fallback device incorrect: {fallback_device} (attendu: cuda:0)")
            
    except Exception as e:
        violations.append(f"❌ Erreur test fonctionnel GPU Manager: {e}")
    
    # 4. Validation GPU si disponible
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
    
    # 5. Rapport
    if violations:
        print("🚫 ÉCHEC VALIDATION:")
        for violation in violations:
            print(f"   {violation}")
        return False
    else:
        print("✅ VALIDATION RÉUSSIE:")
        print("   ✅ utils/gpu_manager.py utilise cuda:0 correctement")
        print("   ✅ Documentation RTX 3090 (CUDA:0) présente")
        print("   ✅ GPU Manager pointe vers RTX 3090")
        if torch.cuda.is_available():
            print("   ✅ RTX 3090 confirmée matériellement")
        return True

if __name__ == "__main__":
    success = validate_gpu_manager_correction()
    if success:
        print("\n🎉 CORRECTION 2 VALIDÉE AVEC SUCCÈS")
        sys.exit(0)
    else:
        print("\n🚫 CORRECTION 2 ÉCHOUÉE")
        sys.exit(1) 