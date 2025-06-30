#!/usr/bin/env python3
"""
Script de test pour UnifiedTTSManager - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Maintenant imports normaux...
import asyncio
import yaml
import time
import logging
from pathlib import Path
from tts_manager import UnifiedTTSManager, validate_rtx3090_configuration

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_unified_tts():
    """Test complet du UnifiedTTSManager avec tous les backends"""
    
    print("🚀 DÉBUT DES TESTS UNIFIEDTTSMANAGER")
    print("=" * 60)
    
    # Validation GPU obligatoire
    validate_rtx3090_configuration()
    
    # Chargement de la configuration
    config_path = Path("config/tts.yaml")
    if not config_path.exists():
        print(f"❌ Fichier de configuration manquant: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialisation du manager
    print("\n📋 INITIALISATION DU MANAGER")
    print("-" * 40)
    manager = UnifiedTTSManager(config)
    
    # Tests de base
    test_phrases = [
        "Bonjour, ceci est un test de synthèse vocale.",
        "SuperWhisper V6 utilise maintenant un système TTS unifié.",
        "Test de performance avec RTX 3090.",
        "Vérification du système de fallback automatique."
    ]
    
    print("\n🎯 TESTS DE SYNTHÈSE")
    print("-" * 40)
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n📝 Test {i}/4: '{phrase[:30]}...'")
        
        start_time = time.perf_counter()
        result = await manager.synthesize(phrase)
        total_time = (time.perf_counter() - start_time) * 1000
        
        if result.success:
            print(f"✅ Succès - Backend: {result.backend_used}")
            print(f"⏱️  Latence: {result.latency_ms:.1f}ms (Total: {total_time:.1f}ms)")
            print(f"📊 Données audio: {len(result.audio_data) if result.audio_data else 0} bytes")
        else:
            print(f"❌ Échec - Erreur: {result.error}")
    
    # Test du cache
    print("\n💾 TEST DU CACHE")
    print("-" * 40)
    
    test_phrase = "Test de cache TTS SuperWhisper V6"
    
    # Premier appel (sans cache)
    print("🔄 Premier appel (sans cache)...")
    result1 = await manager.synthesize(test_phrase)
    if result1.success:
        print(f"✅ Backend utilisé: {result1.backend_used}, Latence: {result1.latency_ms:.1f}ms")
    
    # Deuxième appel (avec cache)
    print("🔄 Deuxième appel (avec cache)...")
    result2 = await manager.synthesize(test_phrase)
    if result2.success:
        print(f"✅ Backend utilisé: {result2.backend_used}, Latence: {result2.latency_ms:.1f}ms")
        if result2.backend_used == "cache":
            print("🎉 Cache fonctionnel !")
    
    # Test de performance
    print("\n⚡ TEST DE PERFORMANCE")
    print("-" * 40)
    
    performance_phrase = "Test de performance TTS avec mesure de latence précise."
    latencies = []
    
    for i in range(5):
        start = time.perf_counter()
        result = await manager.synthesize(performance_phrase, reuse_cache=False)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        if result.success:
            print(f"Test {i+1}: {latency:.1f}ms ({result.backend_used})")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\n📊 STATISTIQUES DE PERFORMANCE:")
        print(f"   Latence moyenne: {avg_latency:.1f}ms")
        print(f"   Latence minimale: {min_latency:.1f}ms")
        print(f"   Latence maximale: {max_latency:.1f}ms")
        
        # Validation des objectifs
        if avg_latency < 120:
            print("🎯 ✅ Objectif <120ms ATTEINT !")
        elif avg_latency < 1000:
            print("🎯 ⚠️ Objectif <120ms manqué, mais <1000ms acceptable")
        else:
            print("🎯 ❌ Performance insuffisante (>1000ms)")
    
    print("\n" + "=" * 60)
    print("🏁 TESTS TERMINÉS")

def main():
    """Point d'entrée principal"""
    try:
        asyncio.run(test_unified_tts())
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 