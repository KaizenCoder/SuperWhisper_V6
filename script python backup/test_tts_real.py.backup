#!/usr/bin/env python3
"""
Script de test pratique pour validation manuelle pendant l'implémentation.
Génère des fichiers audio réels pour écoute et validation.
"""

import asyncio
import time
import yaml
from pathlib import Path
import os
import sys

# Configuration GPU RTX 3090 obligatoire
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Import du manager unifié
sys.path.append('.')
from TTS.tts_manager import UnifiedTTSManager

async def test_real_tts():
    # Chargement config
    with open('config/tts.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    manager = UnifiedTTSManager(config)
    
    # Tests réels avec phrases françaises
    test_phrases = [
        "Bonjour, je suis votre assistant vocal SuperWhisper.",
        "La synthèse vocale fonctionne parfaitement avec RTX 3090.",
        "Test de performance et de qualité audio en français.",
        "Validation du fallback automatique en cas d'erreur."
    ]
    
    print("🎤 TESTS TTS RÉELS - Génération fichiers audio")
    print("=" * 60)
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n📝 Test {i}/4: '{phrase[:30]}...'")
        
        start_time = time.perf_counter()
        result = await manager.synthesize(phrase)
        latency = (time.perf_counter() - start_time) * 1000
        
        if result.success:
            # Sauvegarder audio pour écoute
            audio_file = f"test_output/test_{i}_{result.backend_used}.wav"
            Path("test_output").mkdir(exist_ok=True)
            
            with open(audio_file, 'wb') as f:
                f.write(result.audio_data)
            
            print(f"✅ Backend: {result.backend_used}")
            print(f"✅ Latence: {result.latency_ms:.0f}ms (mesurée: {latency:.0f}ms)")
            print(f"✅ Audio: {audio_file} ({len(result.audio_data)} bytes)")
            print(f"🎧 ÉCOUTER: start {audio_file}")
        else:
            print(f"❌ ÉCHEC: {result.error}")
    
    print(f"\n🎯 VALIDATION MANUELLE:")
    print(f"1. Écouter les 4 fichiers dans test_output/")
    print(f"2. Vérifier qualité audio française")
    print(f"3. Confirmer latence <120ms pour piper_native")
    print(f"4. Tester fallback en désactivant handlers")

if __name__ == "__main__":
    asyncio.run(test_real_tts()) 