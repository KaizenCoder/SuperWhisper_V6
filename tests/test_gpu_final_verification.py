#!/usr/bin/env python3
"""
Vérification finale de la configuration GPU
"""

import os
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