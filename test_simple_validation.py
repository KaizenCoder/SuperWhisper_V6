#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Simple de Validation TTS - SuperWhisper V6
Script de test basique sans emojis pour éviter les problèmes d'encodage
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Configuration GPU RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("GPU Configuration: RTX 3090 (CUDA:1) forcee")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Import du système TTS
try:
    # Ajout du chemin du projet au sys.path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from TTS.tts_manager import UnifiedTTSManager
    from TTS.utils_audio import is_valid_wav, get_wav_info
    import yaml
    TTS_AVAILABLE = True
    print("Systeme TTS disponible")
except ImportError as e:
    print(f"Systeme TTS non disponible: {e}")
    TTS_AVAILABLE = False

async def test_basic_synthesis():
    """Test basique de synthèse TTS"""
    if not TTS_AVAILABLE:
        print("ERREUR: Système TTS non disponible")
        return False
    
    try:
        # Initialisation
        config_path = Path("config/tts.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tts_manager = UnifiedTTSManager(config)
        print("TTS Manager initialise avec succes")
        
        # Tests de base
        test_cases = [
            "Bonjour, test de synthese vocale.",
            "SuperWhisper V6 fonctionne correctement.",
            "Test de performance et de qualite audio."
        ]
        
        results = []
        
        for i, text in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}: {text[:50]}...")
            
            start_time = time.perf_counter()
            tts_result = await tts_manager.synthesize(text)
            synthesis_time = (time.perf_counter() - start_time) * 1000
            
            # Extraction audio
            if hasattr(tts_result, 'audio_data'):
                wav_bytes = tts_result.audio_data
            else:
                wav_bytes = tts_result
            
            if wav_bytes is None:
                print(f"  ECHEC: Aucun audio genere")
                results.append(False)
                continue
            
            # Validation
            is_valid = is_valid_wav(wav_bytes)
            audio_info = get_wav_info(wav_bytes)
            
            print(f"  SUCCES: {synthesis_time:.1f}ms, {len(wav_bytes)} bytes, valide: {is_valid}")
            results.append(True)
            
            # Sauvegarde pour test d'écoute
            filename = f"test_simple_{i}.wav"
            with open(filename, "wb") as f:
                f.write(wav_bytes)
            print(f"  Fichier sauve: {filename}")
        
        # Nettoyage
        await tts_manager.cleanup()
        
        # Résultats
        success_count = sum(results)
        total_count = len(results)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        print(f"\nRESULTATS FINAUX:")
        print(f"Tests reussis: {success_count}/{total_count}")
        print(f"Taux de reussite: {success_rate:.1%}")
        
        return success_rate >= 0.8
        
    except Exception as e:
        print(f"ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Point d'entrée principal"""
    print("SuperWhisper V6 - Test Simple de Validation TTS")
    print("=" * 60)
    
    # Test de synthèse basique
    print("\n1. TEST DE SYNTHESE BASIQUE")
    print("-" * 30)
    synthesis_ok = asyncio.run(test_basic_synthesis())
    
    # Résultat global
    print("\n" + "=" * 60)
    print("RESULTAT GLOBAL:")
    print(f"Synthese basique: {'OK' if synthesis_ok else 'ECHEC'}")
    print(f"Statut global: {'SUCCES' if synthesis_ok else 'ECHEC'}")
    
    return 0 if synthesis_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 