#!/usr/bin/env python3
"""
Monitoring Phase 3 DEMO - SuperWhisper V6 TTS
Démonstration courte (1 minute) du monitoring en temps réel
🚀 Validation rapide des optimisations Phase 3
"""

import os
import sys
import asyncio
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Configuration du logging
logging.basicConfig(
    level=logging.WARNING,  # Moins verbeux pour la démo
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import du système TTS
try:
    from TTS.tts_manager import UnifiedTTSManager
    import yaml
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Système TTS non disponible: {e}")
    TTS_AVAILABLE = False

async def demo_monitoring():
    """Démonstration courte du monitoring Phase 3"""
    print("\n" + "="*80)
    print("🚀 DEMO MONITORING PHASE 3 - 1 MINUTE")
    print("="*80)
    
    if not TTS_AVAILABLE:
        print("❌ Système TTS non disponible")
        return
    
    # Métriques
    metrics = {
        'synthesis_times': [],
        'cache_hits': 0,
        'cache_misses': 0,
        'backend_usage': {},
        'errors': []
    }
    
    try:
        # Initialisation du TTS Manager
        print("🔧 Initialisation TTS Manager...")
        config_path = Path("config/tts.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        start_time = time.perf_counter()
        tts_manager = UnifiedTTSManager(config)
        init_time = (time.perf_counter() - start_time) * 1000
        print(f"✅ TTS Manager initialisé en {init_time:.1f}ms")
        
        # Textes de test pour la démo
        test_texts = [
            "Bonjour, test de performance.",
            "SuperWhisper V6 Phase 3 optimisé.",
            "Cache test - message récurrent.",
            "Cache test - message récurrent.",  # Pour tester le cache
            "Texte long pour tester le chunking intelligent et les performances du système TTS optimisé.",
        ]
        
        print(f"\n🚀 Démarrage monitoring démo (1 minute)")
        print("📊 Tests en cours...\n")
        
        start_monitoring = time.time()
        test_count = 0
        
        # Boucle de monitoring (1 minute)
        while (time.time() - start_monitoring) < 60:  # 1 minute
            try:
                # Sélection du texte
                text = test_texts[test_count % len(test_texts)]
                test_count += 1
                
                # Test de synthèse
                start_synthesis = time.perf_counter()
                tts_result = await tts_manager.synthesize(text)
                synthesis_time = (time.perf_counter() - start_synthesis) * 1000
                
                # Collecte des métriques
                metrics['synthesis_times'].append(synthesis_time)
                
                # Détection cache hit
                if synthesis_time < 10:
                    metrics['cache_hits'] += 1
                    cache_status = "💾 HIT"
                else:
                    metrics['cache_misses'] += 1
                    cache_status = "🔄 MISS"
                
                # Backend estimé
                if synthesis_time < 10:
                    backend = "cache"
                elif synthesis_time < 500:
                    backend = "piper_native"
                elif synthesis_time < 1500:
                    backend = "piper_cli"
                else:
                    backend = "sapi_french"
                
                if backend in metrics['backend_usage']:
                    metrics['backend_usage'][backend] += 1
                else:
                    metrics['backend_usage'][backend] = 1
                
                # Affichage temps réel
                print(f"Test #{test_count:2d}: {synthesis_time:6.1f}ms | {len(text):3d} chars | {cache_status} | {backend}")
                
                # Pause entre tests
                await asyncio.sleep(1)
                
            except Exception as e:
                metrics['errors'].append(str(e))
                print(f"⚠️ Erreur test #{test_count}: {e}")
                await asyncio.sleep(0.5)
        
        # Rapport final
        print("\n" + "="*80)
        print("📊 RAPPORT DEMO MONITORING PHASE 3")
        print("="*80)
        
        total_tests = len(metrics['synthesis_times'])
        total_requests = metrics['cache_hits'] + metrics['cache_misses']
        
        print(f"🧪 Tests effectués: {total_tests}")
        print(f"⏱️ Durée: 1 minute")
        
        if metrics['synthesis_times']:
            avg_time = sum(metrics['synthesis_times']) / len(metrics['synthesis_times'])
            min_time = min(metrics['synthesis_times'])
            max_time = max(metrics['synthesis_times'])
            
            print(f"\n⚡ PERFORMANCES:")
            print(f"   Latence moyenne: {avg_time:.1f}ms")
            print(f"   Latence min/max: {min_time:.1f}ms / {max_time:.1f}ms")
        
        if total_requests > 0:
            hit_rate = (metrics['cache_hits'] / total_requests) * 100
            print(f"\n💾 CACHE:")
            print(f"   Taux de hit: {hit_rate:.1f}%")
            print(f"   Hits/Misses: {metrics['cache_hits']}/{metrics['cache_misses']}")
        
        if metrics['backend_usage']:
            print(f"\n🔧 BACKENDS:")
            total_backend_calls = sum(metrics['backend_usage'].values())
            for backend, count in metrics['backend_usage'].items():
                percentage = (count / total_backend_calls) * 100
                print(f"   {backend}: {count} ({percentage:.1f}%)")
        
        if metrics['errors']:
            print(f"\n⚠️ ERREURS: {len(metrics['errors'])}")
        
        # Validation objectifs
        print(f"\n🎯 OBJECTIFS PHASE 3:")
        
        if metrics['synthesis_times']:
            latency_ok = avg_time < 1000
            print(f"   Latence <1s: {'✅' if latency_ok else '⚠️'} ({avg_time:.1f}ms)")
        
        cache_ok = hit_rate > 10 if total_requests > 0 else False
        print(f"   Cache efficace: {'✅' if cache_ok else '⚠️'} ({hit_rate:.1f}%)")
        
        stability_ok = len(metrics['errors']) == 0
        print(f"   Stabilité: {'✅' if stability_ok else '⚠️'} ({len(metrics['errors'])} erreurs)")
        
        print("\n🎉 Démo monitoring Phase 3 terminée!")
        
        # Nettoyage
        await tts_manager.cleanup()
        
    except Exception as e:
        print(f"❌ Erreur démo monitoring: {e}")
        logging.exception("Erreur détaillée:")

async def main():
    """Point d'entrée principal"""
    print("📊 SuperWhisper V6 - Demo Monitoring Phase 3")
    print("🚀 Validation rapide des performances TTS (1 minute)")
    
    await demo_monitoring()

if __name__ == "__main__":
    asyncio.run(main()) 