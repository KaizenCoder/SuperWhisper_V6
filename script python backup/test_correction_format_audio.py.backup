#!/usr/bin/env python3
"""
Test immédiat des corrections format audio - SuperWhisper V6 TTS
Valide que les fichiers Piper génèrent maintenant des WAV valides
"""

import os
import sys
import asyncio
import yaml
import time
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Import du système TTS
from TTS.tts_manager import UnifiedTTSManager
from TTS.utils_audio import is_valid_wav, get_wav_info

async def test_correction_format_audio():
    """Test principal des corrections format audio"""
    print("\n🔧 TEST CORRECTION FORMAT AUDIO - SUPERWHISPER V6")
    print("=" * 60)
    
    # Chargement configuration
    config_path = Path("config/tts.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialisation manager
    print("📋 Initialisation UnifiedTTSManager...")
    manager = UnifiedTTSManager(config)
    
    # Texte de test
    test_text = "Bonjour, ceci est un test de validation du format audio après correction."
    
    print(f"\n🎯 Texte de test: '{test_text}'")
    print("\n" + "=" * 60)
    
    # Test de chaque backend individuellement
    results = {}
    
    for backend_type, handler in manager.handlers.items():
        backend_name = backend_type.value
        print(f"\n🧪 TEST BACKEND: {backend_name.upper()}")
        print("-" * 40)
        
        try:
            start_time = time.perf_counter()
            
            # Synthèse directe
            audio_data = await handler.synthesize(test_text)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Validation format
            is_valid = is_valid_wav(audio_data)
            wav_info = get_wav_info(audio_data)
            
            results[backend_name] = {
                'success': True,
                'latency_ms': latency_ms,
                'is_valid_wav': is_valid,
                'wav_info': wav_info,
                'size_bytes': len(audio_data)
            }
            
            # Affichage résultats
            status = "✅ VALIDE" if is_valid else "❌ INVALIDE"
            print(f"Format WAV: {status}")
            print(f"Latence: {latency_ms:.1f}ms")
            print(f"Taille: {len(audio_data)} bytes")
            
            if is_valid and 'error' not in wav_info:
                print(f"Canaux: {wav_info.get('channels', 'N/A')}")
                print(f"Fréquence: {wav_info.get('framerate', 'N/A')} Hz")
                print(f"Durée: {wav_info.get('duration_ms', 'N/A')}ms")
            elif 'error' in wav_info:
                print(f"Erreur analyse: {wav_info['error']}")
            
            # Sauvegarde fichier test
            output_file = f"test_output/correction_{backend_name}.wav"
            os.makedirs("test_output", exist_ok=True)
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"Fichier sauvé: {output_file}")
            
        except Exception as e:
            results[backend_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ ERREUR: {e}")
    
    # Test manager unifié
    print(f"\n🎯 TEST MANAGER UNIFIÉ")
    print("-" * 40)
    
    try:
        start_time = time.perf_counter()
        result = await manager.synthesize(test_text)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if result.success:
            is_valid = is_valid_wav(result.audio_data)
            wav_info = get_wav_info(result.audio_data)
            
            status = "✅ VALIDE" if is_valid else "❌ INVALIDE"
            print(f"Backend utilisé: {result.backend_used}")
            print(f"Format WAV: {status}")
            print(f"Latence: {latency_ms:.1f}ms")
            print(f"Taille: {len(result.audio_data)} bytes")
            
            # Sauvegarde fichier principal
            output_file = "test_output/correction_manager_unifie.wav"
            with open(output_file, 'wb') as f:
                f.write(result.audio_data)
            print(f"Fichier sauvé: {output_file}")
            
        else:
            print(f"❌ ERREUR: {result.error}")
    
    except Exception as e:
        print(f"❌ ERREUR MANAGER: {e}")
    
    # Résumé final
    print(f"\n📊 RÉSUMÉ DES CORRECTIONS")
    print("=" * 60)
    
    valid_backends = 0
    total_backends = len(results)
    
    for backend_name, result in results.items():
        if result.get('success', False):
            status = "✅ VALIDE" if result.get('is_valid_wav', False) else "❌ INVALIDE"
            latency = result.get('latency_ms', 0)
            print(f"{backend_name:20} | {status} | {latency:6.1f}ms")
            if result.get('is_valid_wav', False):
                valid_backends += 1
        else:
            print(f"{backend_name:20} | ❌ ERREUR | {result.get('error', 'Inconnue')}")
    
    print("-" * 60)
    print(f"Backends WAV valides: {valid_backends}/{total_backends}")
    
    if valid_backends == total_backends:
        print("🎉 SUCCÈS: Toutes les corrections sont opérationnelles!")
    elif valid_backends > 0:
        print("⚠️  PARTIEL: Certaines corrections fonctionnent")
    else:
        print("❌ ÉCHEC: Aucune correction n'a fonctionné")
    
    # Nettoyage
    await manager.cleanup()
    
    return valid_backends == total_backends

async def test_fichiers_existants():
    """Test des fichiers existants pour comparaison"""
    print(f"\n🔍 ANALYSE FICHIERS EXISTANTS")
    print("=" * 60)
    
    test_dir = Path("test_output")
    if not test_dir.exists():
        print("❌ Répertoire test_output introuvable")
        return
    
    wav_files = list(test_dir.glob("*.wav"))
    if not wav_files:
        print("❌ Aucun fichier WAV trouvé")
        return
    
    print(f"📁 Analyse de {len(wav_files)} fichiers WAV...")
    
    valid_count = 0
    invalid_count = 0
    
    for wav_file in sorted(wav_files):
        try:
            with open(wav_file, 'rb') as f:
                data = f.read()
            
            is_valid = is_valid_wav(data)
            wav_info = get_wav_info(data)
            
            status = "✅ VALIDE" if is_valid else "❌ INVALIDE"
            size_mb = len(data) / 1024 / 1024
            
            print(f"{wav_file.name:30} | {status} | {size_mb:6.2f}MB")
            
            if is_valid:
                valid_count += 1
                if 'error' not in wav_info:
                    duration = wav_info.get('duration_ms', 0)
                    print(f"{'':32} | Durée: {duration}ms")
            else:
                invalid_count += 1
                
        except Exception as e:
            print(f"{wav_file.name:30} | ❌ ERREUR | {e}")
            invalid_count += 1
    
    print("-" * 60)
    print(f"Fichiers valides: {valid_count}")
    print(f"Fichiers invalides: {invalid_count}")

if __name__ == "__main__":
    async def main():
        print("🚀 DÉMARRAGE TEST CORRECTION FORMAT AUDIO")
        
        # Test des corrections
        success = await test_correction_format_audio()
        
        # Test des fichiers existants
        await test_fichiers_existants()
        
        print(f"\n🏁 TEST TERMINÉ")
        if success:
            print("✅ Les corrections de format audio sont opérationnelles!")
        else:
            print("❌ Des problèmes persistent avec les corrections")
    
    asyncio.run(main()) 