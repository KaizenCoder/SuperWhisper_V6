#!/usr/bin/env python3
"""
PRE-FLIGHT CHECK PIPELINE - VALIDATION GPU RTX 3090 OBLIGATOIRE
================================================================
Script de validation GPU obligatoire avant démarrage pipeline SuperWhisper V6

EXIGENCES CRITIQUES :
- RTX 3090 (24GB VRAM) sur CUDA:1 EXCLUSIVEMENT
- CUDA_VISIBLE_DEVICES='1' obligatoire
- Validation mémoire GPU > 20GB
- Aucune utilisation RTX 5060 autorisée

Usage: python PIPELINE/scripts/assert_gpu_env_safe.py
"""

import os
import sys
import logging
from typing import Dict, Any

# Configuration RTX 3090 obligatoire AVANT import torch
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')        # RTX 3090 Bus PCI 1
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')  # Ordre physique stable
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:1024')

# Configuration logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("assert_gpu_env")

def validate_rtx3090_mandatory() -> Dict[str, Any]:
    """
    Validation RTX 3090 obligatoire pour pipeline SuperWhisper V6
    
    Returns:
        Dict contenant les résultats de validation
        
    Raises:
        RuntimeError: Si configuration GPU non conforme
    """
    validation_results = {
        'cuda_available': False,
        'cuda_devices_env': None,
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'device_count': 0,
        'valid_rtx3090': False
    }
    
    try:
        # 1. Vérification CUDA disponible
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("ERREUR: CUDA non disponible - RTX 3090 requise pour pipeline")
        validation_results['cuda_available'] = True
        logger.info("OK: CUDA disponible")
        
        # 2. Vérification CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        validation_results['cuda_devices_env'] = cuda_devices
        if cuda_devices != '1':
            raise RuntimeError(f"ERREUR: CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1' (RTX 3090)")
        logger.info("OK: CUDA_VISIBLE_DEVICES='1' configuré correctement")
        
        # 3. Vérification GPU actuel
        device_count = torch.cuda.device_count()
        validation_results['device_count'] = device_count
        if device_count == 0:
            raise RuntimeError("ERREUR: Aucun GPU CUDA détecté")
        
        # 4. Validation RTX 3090
        gpu_props = torch.cuda.get_device_properties(0)  # device 0 = RTX 3090 via CUDA_VISIBLE_DEVICES='1'
        gpu_name = gpu_props.name
        gpu_memory_gb = gpu_props.total_memory / 1024**3
        
        validation_results['gpu_name'] = gpu_name
        validation_results['gpu_memory_gb'] = gpu_memory_gb
        
        # Validation mémoire RTX 3090 (>20GB)
        if gpu_memory_gb < 20:
            raise RuntimeError(f"ERREUR: GPU '{gpu_name}' ({gpu_memory_gb:.1f}GB) insuffisante - RTX 3090 (24GB) requise")
        
        # Validation nom RTX 3090
        if "3090" not in gpu_name and "RTX" in gpu_name:
            logger.warning(f"ATTENTION: GPU '{gpu_name}' détectée - vérifier qu'il s'agit bien de RTX 3090")
        
        validation_results['valid_rtx3090'] = True
        logger.info(f"OK: RTX 3090 validée: {gpu_name} ({gpu_memory_gb:.1f}GB VRAM)")
        
        # 5. Informations supplémentaires
        logger.info(f"GPU Memory: {gpu_memory_gb:.1f}GB / Devices: {device_count}")
        logger.info(f"GPU Mapping: cuda:0 -> {gpu_name} (Bus PCI 1)")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"ERREUR validation GPU: {e}")
        validation_results['error'] = str(e)
        raise

def main():
    """Point d'entrée principal du script de validation GPU"""
    logger.info("DEMARRAGE VALIDATION GPU RTX 3090 PIPELINE...")
    
    try:
        results = validate_rtx3090_mandatory()
        
        # Affichage résultats
        print("\n" + "="*60)
        print("RESULTATS VALIDATION GPU RTX 3090 PIPELINE")
        print("="*60)
        print(f"OK CUDA Disponible: {results['cuda_available']}")
        print(f"OK CUDA_VISIBLE_DEVICES: '{results['cuda_devices_env']}'")
        print(f"OK GPU Detectee: {results['gpu_name']}")
        print(f"OK Memoire GPU: {results['gpu_memory_gb']:.1f}GB")
        print(f"OK RTX 3090 Validee: {results['valid_rtx3090']}")
        print("="*60)
        print("PIPELINE AUTORISE - Configuration GPU RTX 3090 conforme")
        print("="*60)
        
        return 0
        
    except RuntimeError as e:
        print("\n" + "="*60)
        print("ECHEC VALIDATION GPU RTX 3090 PIPELINE")
        print("="*60)
        print(f"ERREUR: {e}")
        print("ACTIONS REQUISES:")
        print("   - Verifier configuration dual-GPU")
        print("   - Configurer CUDA_VISIBLE_DEVICES='1'")
        print("   - Valider RTX 3090 sur Bus PCI 1")
        print("="*60)
        print("PIPELINE BLOQUE - Corriger configuration GPU avant continuation")
        print("="*60)
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 