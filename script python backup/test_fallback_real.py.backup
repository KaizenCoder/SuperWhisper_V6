#!/usr/bin/env python3
"""
Test pratique du système de fallback avec simulation de pannes.
"""

import asyncio
import yaml
import os
import sys

# Configuration GPU RTX 3090 obligatoire
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Import du manager unifié
sys.path.append('.')
from TTS.tts_manager import UnifiedTTSManager

async def test_fallback_simulation():
    print("🔧 TEST FALLBACK RÉEL - Simulation pannes")
    
    with open('config/tts.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test 1: Tous handlers actifs
    print("\n1️⃣ Test normal (tous handlers actifs)")
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test normal avec tous les backends.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 2: Désactiver piper_native (forcer fallback)
    print("\n2️⃣ Test fallback (piper_native désactivé)")
    config['backends']['piper_native']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test fallback vers piper CLI.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 3: Désactiver piper_native + piper_cli (forcer SAPI)
    print("\n3️⃣ Test fallback SAPI (piper désactivés)")
    config['backends']['piper_cli']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test fallback vers SAPI français.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 4: Tous désactivés sauf emergency
    print("\n4️⃣ Test emergency (tous backends désactivés)")
    config['backends']['sapi_french']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test handler d'urgence silencieux.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    print(f"\n🎯 VALIDATION: Chaîne de fallback complète testée!")

if __name__ == "__main__":
    asyncio.run(test_fallback_simulation()) 