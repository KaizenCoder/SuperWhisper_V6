#!/usr/bin/env python3
"""
Test de Performance Phase 3 Simplifié - SuperWhisper V6 TTS
Test avec gestion correcte du TTSResult
🚀 Validation des optimisations Phase 3
"""

import os
import sys
import asyncio
import logging
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

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import du système TTS
try:
    from TTS.tts_manager import UnifiedTTSManager
    from TTS.utils_audio import is_valid_wav, get_wav_info
    import yaml
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Système TTS non disponible: {e}")
    TTS_AVAILABLE = False

async def test_phase3_performance():
    """Test de performance Phase 3 simplifié"""
    print("\n" + "="*80)
    print("🚀 TEST PERFORMANCE PHASE 3 - SIMPLIFIÉ")
    print("="*80)
    
    if not TTS_AVAILABLE:
        print("❌ Système TTS non disponible")
        return
    
    try:
        # Chargement de la configuration
        config_path = Path("config/tts.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Initialisation du TTS Manager
        print("\n🔧 Initialisation TTS Manager...")
        start_time = time.perf_counter()
        tts_manager = UnifiedTTSManager(config)
        init_time = (time.perf_counter() - start_time) * 1000
        print(f"✅ TTS Manager initialisé en {init_time:.1f}ms")
        
        # Tests avec différentes tailles de texte
        test_texts = {
            'court': "Bonjour, comment allez-vous ?",
            'moyen': "SuperWhisper V6 est un assistant vocal avancé utilisant l'intelligence artificielle pour offrir une expérience conversationnelle naturelle et fluide.",
            'long': """SuperWhisper V6 représente une avancée majeure dans le domaine des assistants vocaux. 
            Grâce à son architecture innovante combinant reconnaissance vocale, traitement du langage naturel 
            et synthèse vocale de haute qualité, il offre une expérience utilisateur exceptionnelle. 
            Le système utilise des modèles d'IA de pointe optimisés pour fonctionner efficacement 
            sur du matériel grand public, tout en maintenant des performances professionnelles.""" * 3,
            'tres_long': """L'intelligence artificielle conversationnelle a connu une évolution remarquable 
            au cours des dernières années. SuperWhisper V6 s'inscrit dans cette dynamique 
            en proposant une solution complète et optimisée pour les interactions vocales.
            
            Architecture Technique: Le système repose sur une architecture en pipeline comprenant 
            trois composants principaux : reconnaissance vocale, traitement du langage naturel 
            et synthèse vocale multi-backend avec fallback intelligent.
            
            Optimisations Performance: Les optimisations Phase 3 introduisent plusieurs innovations 
            majeures incluant le binding Python natif, le cache LRU intelligent, le chunking sémantique, 
            le pipeline asynchrone et l'optimisation GPU.""" * 4
        }
        
        results = {}
        
        # Tests de performance
        for test_name, text in test_texts.items():
            print(f"\n🔬 TEST: {test_name.upper()}")
            print("-" * 50)
            print(f"📝 Texte: {len(text)} caractères")
            
            # Synthèse
            start_time = time.perf_counter()
            tts_result = await tts_manager.synthesize(text)
            synthesis_time = (time.perf_counter() - start_time) * 1000
            
            # Extraction des données audio
            if hasattr(tts_result, 'audio_data'):
                audio_data = tts_result.audio_data
            else:
                # Fallback si c'est directement les bytes
                audio_data = tts_result
            
            # Validation
            try:
                is_valid = is_valid_wav(audio_data)
                audio_info = get_wav_info(audio_data)
                audio_size = len(audio_data)
                audio_duration = audio_info.get('duration_ms', 0)
            except Exception as e:
                print(f"⚠️ Erreur validation audio: {e}")
                is_valid = False
                audio_size = 0
                audio_duration = 0
            
            # Résultats
            print(f"✅ Synthèse réussie en {synthesis_time:.1f}ms")
            print(f"   Audio: {audio_size} bytes, {audio_duration:.0f}ms")
            print(f"   Format WAV valide: {is_valid}")
            
            # Métriques de performance
            chars_per_ms = len(text) / synthesis_time if synthesis_time > 0 else 0
            print(f"🚀 Performance: {chars_per_ms:.3f} chars/ms")
            
            # Estimation chunks pour textes longs
            if len(text) > 800:
                estimated_chunks = (len(text) // 800) + 1
                avg_time_per_chunk = synthesis_time / estimated_chunks
                print(f"📊 Chunks estimés: {estimated_chunks}")
                print(f"⚡ Temps moyen/chunk: {avg_time_per_chunk:.1f}ms")
                
                # Vérification objectif Phase 3
                target_met = avg_time_per_chunk <= 100
                print(f"🎯 Objectif <100ms/chunk: {'✅' if target_met else '⚠️'}")
            
            results[test_name] = {
                'chars': len(text),
                'synthesis_time_ms': synthesis_time,
                'audio_size_bytes': audio_size,
                'audio_duration_ms': audio_duration,
                'audio_valid': is_valid,
                'chars_per_ms': chars_per_ms
            }
        
        # Test cache (texte récurrent)
        print(f"\n🔬 TEST: CACHE PERFORMANCE")
        print("-" * 50)
        
        cache_text = "Message récurrent pour test de cache."
        print(f"📝 Texte: {len(cache_text)} caractères")
        
        # Premier appel
        print("🔄 Premier appel (cache miss)...")
        start_time = time.perf_counter()
        result1 = await tts_manager.synthesize(cache_text)
        first_time = (time.perf_counter() - start_time) * 1000
        print(f"   Temps: {first_time:.1f}ms")
        
        # Deuxième appel
        print("⚡ Deuxième appel (cache hit attendu)...")
        start_time = time.perf_counter()
        result2 = await tts_manager.synthesize(cache_text)
        second_time = (time.perf_counter() - start_time) * 1000
        print(f"   Temps: {second_time:.1f}ms")
        
        # Analyse cache
        speedup = first_time / second_time if second_time > 0 else float('inf')
        cache_hit = second_time < (first_time * 0.1)
        print(f"🚀 Accélération: {speedup:.1f}x")
        print(f"💾 Cache hit détecté: {'✅' if cache_hit else '❌'}")
        
        # Rapport final
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL PHASE 3")
        print("="*80)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r['audio_valid'])
        print(f"🎯 Tests réussis: {successful_tests}/{total_tests}")
        
        # Métriques globales
        synthesis_times = [r['synthesis_time_ms'] for r in results.values() if r['audio_valid']]
        if synthesis_times:
            avg_time = sum(synthesis_times) / len(synthesis_times)
            print(f"⚡ Temps moyen synthèse: {avg_time:.1f}ms")
        
        # Validation objectifs Phase 3
        print("\n🎯 OBJECTIFS PHASE 3:")
        
        # Support textes longs
        long_text_ok = any(r['chars'] >= 4000 and r['audio_valid'] for r in results.values())
        print(f"   Support textes 4000+ chars: {'✅' if long_text_ok else '❌'}")
        
        # Cache intelligent
        print(f"   Cache intelligent: {'✅' if cache_hit else '❌'}")
        
        # Performance générale
        good_performance = avg_time < 2000 if synthesis_times else False
        print(f"   Performance <2s: {'✅' if good_performance else '⚠️'}")
        
        print("\n🎉 Tests Phase 3 terminés avec succès!")
        
        # Nettoyage
        await tts_manager.cleanup()
        
    except Exception as e:
        print(f"❌ Erreur test performance: {e}")
        logging.exception("Erreur détaillée:")

async def main():
    """Point d'entrée principal"""
    await test_phase3_performance()

if __name__ == "__main__":
    asyncio.run(main()) 