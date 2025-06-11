#!/usr/bin/env python3
"""
VALIDATION CORRECTION 1 : tests/test_stt_handler.py
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

def validate_stt_handler_correction():
    """Validation que le fichier corrigé utilise RTX 3090 (cuda:0)"""
    print("🔍 VALIDATION CORRECTION - tests/test_stt_handler.py")
    
    # 1. Vérifier le contenu du fichier
    with open('tests/test_stt_handler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 2. Vérifications critiques
    violations = []
    
    # Vérifier qu'il n'y a plus de cuda:1
    if 'cuda:1' in content and 'RTX 5060' not in content:
        violations.append("❌ cuda:1 trouvé sans documentation RTX 5060")
    
    # Vérifier que cuda:0 est utilisé
    if 'cuda:0' not in content:
        violations.append("❌ cuda:0 non trouvé - RTX 3090 non configurée")
    
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
        print("   ✅ tests/test_stt_handler.py utilise cuda:0 correctement")
        print("   ✅ Documentation RTX 3090 (CUDA:0) présente")
        print("   ✅ Aucune référence cuda:1 non documentée")
        if torch.cuda.is_available():
            print("   ✅ RTX 3090 confirmée matériellement")
        return True

if __name__ == "__main__":
    success = validate_stt_handler_correction()
    if success:
        print("\n🎉 CORRECTION 1 VALIDÉE AVEC SUCCÈS")
        sys.exit(0)
    else:
        print("\n🚫 CORRECTION 1 ÉCHOUÉE")
        sys.exit(1) 