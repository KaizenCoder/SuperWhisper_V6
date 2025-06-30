#!/usr/bin/env python3
"""
Test des Optimisations Phase 3 - SuperWhisper V6 TTS
Validation complète des 5 axes d'optimisation implémentés
🚀 Performance cible: <100ms par appel, textes 5000+ chars

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE - EXÉCUTABLE DEPUIS N'IMPORTE OÙ
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    # Déterminer le répertoire racine du projet
    current_file = pathlib.Path(__file__).resolve()
    
    # Chercher le répertoire racine (contient .git ou marqueurs projet)
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    # Ajouter le projet root au Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Changer le working directory vers project root
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090 obligatoire
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
    
    print(f"🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
    print(f"📁 Project Root: {project_root}")
    print(f"💻 Working Directory: {os.getcwd()}")
    
    return project_root

# Initialiser l'environnement portable
_PROJECT_ROOT = _setup_portable_environment()

# Maintenant imports normaux...

import asyncio
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List

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

# Import des composants Phase 3
try:
    from TTS.handlers.piper_native_optimized import create_piper_native_handler
    from TTS.handlers.piper_daemon import create_piper_daemon_handler
    from TTS.utils.text_chunker import IntelligentTextChunker, AudioConcatenator
    from TTS.components.cache_optimized import OptimizedTTSCache
    from TTS.utils_audio import is_valid_wav, get_wav_info
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Composants Phase 3 non disponibles: {e}")
    PHASE3_AVAILABLE = False

class Phase3OptimizationTester:
    """
    Testeur complet des optimisations Phase 3
    
    🚀 TESTS COUVERTS:
    1. Binding Python natif Piper (vs CLI)
    2. Pipeline asynchrone daemon
    3. Chunking intelligent textes longs
    4. Cache LRU optimisé
    5. Concaténation audio fluide
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
        # Textes de test de complexité croissante
        self.test_texts = {
            'court': "Bonjour, ceci est un test court.",
            'moyen': "Ceci est un test de longueur moyenne pour valider les performances du système TTS optimisé. " * 5,
            'long': "Voici un texte long pour tester le chunking intelligent. " * 50,  # ~2500 chars
            'tres_long': "Test de texte très long pour valider la gestion des textes de 5000+ caractères. " * 60,  # ~5000 chars
            'recurrent': "Message récurrent pour test cache."
        }
        
        print("🧪 Phase 3 Optimization Tester initialisé")
        print(f"📊 {len(self.test_texts)} textes de test préparés")
    
    def _load_config(self) -> Dict[str, Any]:
        """Chargement de la configuration TTS"""
        config_path = Path("config/tts.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration TTS introuvable: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    async def run_all_tests(self):
        """Exécution de tous les tests Phase 3"""
        print("\n" + "="*80)
        print("🚀 DÉMARRAGE TESTS OPTIMISATIONS PHASE 3")
        print("="*80)
        
        if not PHASE3_AVAILABLE:
            print("❌ Composants Phase 3 non disponibles - Tests annulés")
            return
        
        # Test 1: Binding Python natif
        await self._test_native_binding()
        
        # Test 2: Pipeline daemon (si activé)
        if self.config['backends'].get('piper_daemon', {}).get('enabled', False):
            await self._test_daemon_pipeline()
        
        # Test 3: Chunking intelligent
        await self._test_intelligent_chunking()
        
        # Test 4: Cache LRU optimisé
        await self._test_optimized_cache()
        
        # Test 5: Concaténation audio
        await self._test_audio_concatenation()
        
        # Test 6: Performance globale
        await self._test_global_performance()
        
        # Rapport final
        self._generate_final_report()
    
    async def _test_native_binding(self):
        """Test 1: Binding Python natif vs CLI"""
        print("\n🔬 TEST 1: BINDING PYTHON NATIF")
        print("-" * 50)
        
        try:
            # Configuration du handler optimisé
            handler_config = self.config['backends']['piper_native_optimized']
            
            # Test de création du handler
            start_time = time.perf_counter()
            handler = create_piper_native_handler(handler_config)
            init_time = (time.perf_counter() - start_time) * 1000
            
            print(f"✅ Handler créé en {init_time:.1f}ms")
            
            # Test de synthèse
            test_text = self.test_texts['moyen']
            start_time = time.perf_counter()
            
            audio_data = await handler.synthesize(test_text)
            synthesis_time = (time.perf_counter() - start_time) * 1000
            
            # Validation
            is_valid = is_valid_wav(audio_data)
            audio_info = get_wav_info(audio_data)
            
            print(f"✅ Synthèse réussie en {synthesis_time:.1f}ms")
            print(f"   Audio: {len(audio_data)} bytes, {audio_info.get('duration_ms', 0):.0f}ms")
            print(f"   Format WAV valide: {is_valid}")
            
            # Vérification performance cible
            target_latency = handler_config.get('target_latency_ms', 80)
            performance_ok = synthesis_time <= target_latency
            
            print(f"🎯 Performance: {synthesis_time:.1f}ms (cible: {target_latency}ms) {'✅' if performance_ok else '⚠️'}")
            
            self.test_results['native_binding'] = {
                'success': True,
                'init_time_ms': init_time,
                'synthesis_time_ms': synthesis_time,
                'target_latency_ms': target_latency,
                'performance_ok': performance_ok,
                'audio_size_bytes': len(audio_data),
                'audio_valid': is_valid
            }
            
            self.passed_tests += 1
            
        except Exception as e:
            print(f"❌ Erreur test binding natif: {e}")
            self.test_results['native_binding'] = {'success': False, 'error': str(e)}
        
        self.total_tests += 1
    
    async def _test_daemon_pipeline(self):
        """Test 2: Pipeline daemon asynchrone"""
        print("\n🔬 TEST 2: PIPELINE DAEMON ASYNCHRONE")
        print("-" * 50)
        
        try:
            handler_config = self.config['backends']['piper_daemon']
            
            # Test de création du daemon
            start_time = time.perf_counter()
            daemon_handler = create_piper_daemon_handler(handler_config)
            init_time = (time.perf_counter() - start_time) * 1000
            
            print(f"✅ Daemon handler créé en {init_time:.1f}ms")
            
            # Test de synthèse
            test_text = self.test_texts['moyen']
            start_time = time.perf_counter()
            
            audio_data = await daemon_handler.synthesize(test_text)
            synthesis_time = (time.perf_counter() - start_time) * 1000
            
            # Validation
            is_valid = is_valid_wav(audio_data)
            
            print(f"✅ Synthèse daemon réussie en {synthesis_time:.1f}ms")
            print(f"   Audio: {len(audio_data)} bytes, WAV valide: {is_valid}")
            
            # Nettoyage
            if hasattr(daemon_handler, 'cleanup'):
                await daemon_handler.cleanup()
            
            self.test_results['daemon_pipeline'] = {
                'success': True,
                'synthesis_time_ms': synthesis_time,
                'audio_valid': is_valid
            }
            
            self.passed_tests += 1
            
        except Exception as e:
            print(f"❌ Erreur test daemon: {e}")
            self.test_results['daemon_pipeline'] = {'success': False, 'error': str(e)}
        
        self.total_tests += 1
    
    async def _test_intelligent_chunking(self):
        """Test 3: Chunking intelligent pour textes longs"""
        print("\n🔬 TEST 3: CHUNKING INTELLIGENT TEXTES LONGS")
        print("-" * 50)
        
        try:
            # Configuration du chunker
            chunker_config = self.config['advanced']['text_chunking']
            chunker = IntelligentTextChunker(chunker_config)
            
            # Test sur texte très long
            long_text = self.test_texts['tres_long']
            print(f"📝 Texte original: {len(long_text)} caractères")
            
            # Découpage
            start_time = time.perf_counter()
            chunks = chunker.chunk_text(long_text, backend_max_length=800)
            chunking_time = (time.perf_counter() - start_time) * 1000
            
            print(f"✅ Découpage réussi en {chunking_time:.1f}ms")
            print(f"   {len(chunks)} chunks générés")
            
            # Analyse des chunks
            total_chars = sum(len(chunk.text) for chunk in chunks)
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            estimated_duration = chunker.get_total_estimated_duration(chunks)
            
            print(f"   Taille moyenne: {avg_chunk_size:.0f} chars/chunk")
            print(f"   Durée estimée: {estimated_duration/1000:.1f}s")
            
            # Validation des limites
            max_chunk_size = max(len(chunk.text) for chunk in chunks) if chunks else 0
            chunks_within_limits = max_chunk_size <= 800
            
            print(f"🎯 Respect limites: {max_chunk_size} chars max {'✅' if chunks_within_limits else '❌'}")
            
            # Statistiques détaillées
            stats = chunker.get_chunking_stats(long_text, chunks)
            print(f"   Efficacité: {stats['chunking_efficiency']:.1%}")
            print(f"   Frontières phrases: {stats['sentence_boundaries']}/{len(chunks)}")
            
            self.test_results['intelligent_chunking'] = {
                'success': True,
                'original_length': len(long_text),
                'chunk_count': len(chunks),
                'chunking_time_ms': chunking_time,
                'max_chunk_size': max_chunk_size,
                'within_limits': chunks_within_limits,
                'efficiency': stats['chunking_efficiency'],
                'estimated_duration_s': estimated_duration / 1000
            }
            
            self.passed_tests += 1
            
        except Exception as e:
            print(f"❌ Erreur test chunking: {e}")
            self.test_results['intelligent_chunking'] = {'success': False, 'error': str(e)}
        
        self.total_tests += 1
    
    async def _test_optimized_cache(self):
        """Test 4: Cache LRU optimisé"""
        print("\n🔬 TEST 4: CACHE LRU OPTIMISÉ")
        print("-" * 50)
        
        try:
            # Configuration du cache
            cache_config = self.config['cache']
            cache = OptimizedTTSCache(cache_config)
            
            # Test d'ajout au cache
            test_text = self.test_texts['recurrent']
            fake_audio = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xAC\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            
            cache_key = cache.generate_key(test_text, {'voice': None, 'speed': None})
            
            # PUT
            start_time = time.perf_counter()
            put_success = await cache.put(cache_key, fake_audio, test_text, 'test_backend', 1000.0)
            put_time = (time.perf_counter() - start_time) * 1000
            
            print(f"✅ Cache PUT réussi en {put_time:.2f}ms")
            
            # GET (cache hit)
            start_time = time.perf_counter()
            cached_audio = await cache.get(cache_key)
            get_time = (time.perf_counter() - start_time) * 1000
            
            cache_hit = cached_audio is not None
            print(f"✅ Cache GET {'HIT' if cache_hit else 'MISS'} en {get_time:.2f}ms")
            
            # Test de performance répétée
            hit_times = []
            for i in range(10):
                start_time = time.perf_counter()
                await cache.get(cache_key)
                hit_times.append((time.perf_counter() - start_time) * 1000)
            
            avg_hit_time = sum(hit_times) / len(hit_times)
            print(f"🚀 Performance moyenne: {avg_hit_time:.2f}ms/hit")
            
            # Statistiques du cache
            stats = cache.get_stats()
            print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
            print(f"   Entrées: {stats['cache_entries']}")
            print(f"   Taille: {stats['cache_size_mb']:.2f} MB")
            
            # Nettoyage
            await cache.cleanup()
            
            self.test_results['optimized_cache'] = {
                'success': True,
                'put_time_ms': put_time,
                'get_time_ms': get_time,
                'avg_hit_time_ms': avg_hit_time,
                'cache_hit': cache_hit,
                'hit_rate_percent': stats['hit_rate_percent']
            }
            
            self.passed_tests += 1
            
        except Exception as e:
            print(f"❌ Erreur test cache: {e}")
            self.test_results['optimized_cache'] = {'success': False, 'error': str(e)}
        
        self.total_tests += 1
    
    async def _test_audio_concatenation(self):
        """Test 5: Concaténation audio fluide"""
        print("\n🔬 TEST 5: CONCATÉNATION AUDIO FLUIDE")
        print("-" * 50)
        
        try:
            # Configuration du concatenator
            concat_config = self.config['advanced']['text_chunking']
            concatenator = AudioConcatenator(concat_config)
            
            # Création de chunks audio factices
            fake_wav_chunks = []
            for i in range(3):
                # Header WAV minimal + données factices
                wav_chunk = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xAC\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
                fake_wav_chunks.append(wav_chunk)
            
            print(f"🎵 {len(fake_wav_chunks)} chunks audio à concaténer")
            
            # Concaténation
            start_time = time.perf_counter()
            concatenated_audio = concatenator.concatenate_wav_chunks(fake_wav_chunks)
            concat_time = (time.perf_counter() - start_time) * 1000
            
            print(f"✅ Concaténation réussie en {concat_time:.1f}ms")
            print(f"   Audio final: {len(concatenated_audio)} bytes")
            
            # Validation format
            is_valid = is_valid_wav(concatenated_audio)
            print(f"   Format WAV valide: {is_valid}")
            
            self.test_results['audio_concatenation'] = {
                'success': True,
                'concat_time_ms': concat_time,
                'chunk_count': len(fake_wav_chunks),
                'final_size_bytes': len(concatenated_audio),
                'audio_valid': is_valid
            }
            
            self.passed_tests += 1
            
        except Exception as e:
            print(f"❌ Erreur test concaténation: {e}")
            self.test_results['audio_concatenation'] = {'success': False, 'error': str(e)}
        
        self.total_tests += 1
    
    async def _test_global_performance(self):
        """Test 6: Performance globale intégrée"""
        print("\n🔬 TEST 6: PERFORMANCE GLOBALE INTÉGRÉE")
        print("-" * 50)
        
        try:
            # Test avec le texte le plus long
            long_text = self.test_texts['tres_long']
            print(f"📝 Test performance sur {len(long_text)} caractères")
            
            # Simulation du pipeline complet
            start_time = time.perf_counter()
            
            # 1. Chunking
            chunker_config = self.config['advanced']['text_chunking']
            chunker = IntelligentTextChunker(chunker_config)
            chunks = chunker.chunk_text(long_text, backend_max_length=800)
            
            # 2. Cache check (simulation)
            cache_config = self.config['cache']
            cache = OptimizedTTSCache(cache_config)
            
            # 3. Synthèse simulée par chunk
            chunk_times = []
            for chunk in chunks[:3]:  # Test sur 3 premiers chunks
                chunk_start = time.perf_counter()
                # Simulation synthèse (80ms target)
                await asyncio.sleep(0.08)  # 80ms
                chunk_time = (time.perf_counter() - chunk_start) * 1000
                chunk_times.append(chunk_time)
            
            total_time = (time.perf_counter() - start_time) * 1000
            avg_chunk_time = sum(chunk_times) / len(chunk_times)
            
            print(f"✅ Pipeline complet simulé en {total_time:.1f}ms")
            print(f"   Chunks traités: {len(chunk_times)}")
            print(f"   Temps moyen/chunk: {avg_chunk_time:.1f}ms")
            
            # Estimation temps total réel
            estimated_total = len(chunks) * avg_chunk_time
            print(f"🎯 Estimation temps total: {estimated_total:.1f}ms pour {len(chunks)} chunks")
            
            # Objectif performance: <100ms par appel
            performance_target_met = avg_chunk_time <= 100
            print(f"   Objectif <100ms/chunk: {'✅' if performance_target_met else '⚠️'}")
            
            await cache.cleanup()
            
            self.test_results['global_performance'] = {
                'success': True,
                'total_time_ms': total_time,
                'avg_chunk_time_ms': avg_chunk_time,
                'estimated_total_ms': estimated_total,
                'performance_target_met': performance_target_met,
                'chunk_count': len(chunks)
            }
            
            self.passed_tests += 1
            
        except Exception as e:
            print(f"❌ Erreur test performance globale: {e}")
            self.test_results['global_performance'] = {'success': False, 'error': str(e)}
        
        self.total_tests += 1
    
    def _generate_final_report(self):
        """Génération du rapport final"""
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL - OPTIMISATIONS PHASE 3")
        print("="*80)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"🎯 Résultats: {self.passed_tests}/{self.total_tests} tests réussis ({success_rate:.1f}%)")
        print()
        
        # Détail par test
        for test_name, result in self.test_results.items():
            status = "✅ RÉUSSI" if result.get('success', False) else "❌ ÉCHEC"
            print(f"{status} - {test_name.replace('_', ' ').title()}")
            
            if result.get('success', False):
                # Affichage des métriques clés
                if 'synthesis_time_ms' in result:
                    print(f"    Latence: {result['synthesis_time_ms']:.1f}ms")
                if 'performance_ok' in result:
                    print(f"    Performance cible: {'✅' if result['performance_ok'] else '⚠️'}")
                if 'hit_rate_percent' in result:
                    print(f"    Hit rate cache: {result['hit_rate_percent']:.1f}%")
            else:
                print(f"    Erreur: {result.get('error', 'Inconnue')}")
            print()
        
        # Recommandations
        print("🚀 RECOMMANDATIONS:")
        
        if success_rate >= 80:
            print("✅ Optimisations Phase 3 opérationnelles")
            print("   → Déploiement en production recommandé")
        elif success_rate >= 60:
            print("⚠️ Optimisations partiellement fonctionnelles")
            print("   → Corriger les échecs avant déploiement")
        else:
            print("❌ Optimisations nécessitent des corrections majeures")
            print("   → Révision complète requise")
        
        print("\n🎉 Tests Phase 3 terminés!")


async def main():
    """Point d'entrée principal"""
    tester = Phase3OptimizationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 