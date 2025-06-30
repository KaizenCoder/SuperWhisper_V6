#!/usr/bin/env python3
"""
Test de Performance Phase 3 - SuperWhisper V6 TTS
Test réel avec UnifiedTTSManager et texte long (5000+ chars)
🚀 Validation des optimisations en conditions réelles

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
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Système TTS non disponible: {e}")
    TTS_AVAILABLE = False

class Phase3PerformanceTester:
    """
    Testeur de performance Phase 3 en conditions réelles
    
    🚀 TESTS RÉELS:
    1. Texte court (cache miss/hit)
    2. Texte moyen (performance standard)
    3. Texte long (chunking intelligent)
    4. Texte très long (5000+ chars)
    5. Texte récurrent (cache hit)
    """
    
    def __init__(self):
        self.tts_manager = None
        self.test_results = {}
        
        # Textes de test réalistes
        self.test_texts = {
            'court': "Bonjour, comment allez-vous aujourd'hui ?",
            
            'moyen': """
            Bienvenue dans SuperWhisper V6, l'assistant vocal de nouvelle génération. 
            Ce système utilise des technologies avancées d'intelligence artificielle 
            pour vous offrir une expérience conversationnelle naturelle et fluide.
            """,
            
            'long': """
            SuperWhisper V6 représente une avancée majeure dans le domaine des assistants vocaux. 
            Grâce à son architecture innovante combinant reconnaissance vocale, traitement du langage naturel 
            et synthèse vocale de haute qualité, il offre une expérience utilisateur exceptionnelle. 
            
            Le système utilise des modèles d'IA de pointe optimisés pour fonctionner efficacement 
            sur du matériel grand public, tout en maintenant des performances professionnelles. 
            L'architecture modulaire permet une adaptation flexible aux besoins spécifiques 
            de chaque utilisateur et cas d'usage.
            
            Les optimisations Phase 3 apportent des améliorations significatives en termes 
            de latence et de capacité de traitement, permettant de gérer des textes longs 
            avec une fluidité remarquable.
            """ * 3,  # ~2400 chars
            
            'tres_long': """
            L'intelligence artificielle conversationnelle a connu une évolution remarquable 
            au cours des dernières années. SuperWhisper V6 s'inscrit dans cette dynamique 
            en proposant une solution complète et optimisée pour les interactions vocales.
            
            Architecture Technique:
            Le système repose sur une architecture en pipeline comprenant trois composants principaux :
            - Un module de reconnaissance vocale (STT) utilisant des modèles Whisper optimisés
            - Un moteur de traitement du langage naturel basé sur des LLM de dernière génération
            - Un système de synthèse vocale (TTS) multi-backend avec fallback intelligent
            
            Optimisations Performance:
            Les optimisations Phase 3 introduisent plusieurs innovations majeures :
            
            1. Binding Python Natif : Remplacement des appels CLI par des bindings Python directs,
               réduisant la latence de 500ms à moins de 80ms par synthèse.
            
            2. Cache LRU Intelligent : Système de cache avancé permettant une réponse instantanée
               pour les textes récurrents, avec éviction intelligente et métriques détaillées.
            
            3. Chunking Sémantique : Découpage intelligent des textes longs en respectant
               les frontières de phrases et en optimisant la fluidité de la synthèse.
            
            4. Pipeline Asynchrone : Architecture non-bloquante permettant le traitement
               parallèle de multiples requêtes avec une utilisation optimale des ressources.
            
            5. Optimisation GPU : Réaffectation intelligente des GPU pour minimiser
               la contention et maximiser les performances globales du système.
            
            Cas d'Usage:
            SuperWhisper V6 est conçu pour répondre à une large gamme de besoins :
            - Assistants personnels intelligents
            - Systèmes de support client automatisés
            - Outils d'accessibilité pour personnes malvoyantes
            - Applications éducatives interactives
            - Interfaces vocales pour systèmes industriels
            
            La flexibilité de l'architecture permet une adaptation rapide à des domaines
            spécifiques tout en maintenant des performances optimales.
            """ * 2,  # ~5000+ chars
            
            'recurrent': "Message récurrent pour test de cache."
        }
        
        print("🧪 Phase 3 Performance Tester initialisé")
        print(f"📊 {len(self.test_texts)} textes de test préparés")
        
        # Affichage des tailles
        for name, text in self.test_texts.items():
            print(f"   {name}: {len(text)} caractères")
    
    async def run_performance_tests(self):
        """Exécution des tests de performance réels"""
        print("\n" + "="*80)
        print("🚀 TESTS PERFORMANCE PHASE 3 - CONDITIONS RÉELLES")
        print("="*80)
        
        if not TTS_AVAILABLE:
            print("❌ Système TTS non disponible - Tests annulés")
            return
        
        try:
            # Initialisation du TTS Manager
            await self._initialize_tts_manager()
            
            # Tests de performance par complexité croissante
            await self._test_text_court()
            await self._test_text_moyen()
            await self._test_text_long()
            await self._test_text_tres_long()
            await self._test_cache_performance()
            
            # Rapport final
            self._generate_performance_report()
            
        except Exception as e:
            print(f"❌ Erreur tests performance: {e}")
            logging.exception("Erreur détaillée:")
        
        finally:
            # Nettoyage
            if self.tts_manager:
                await self.tts_manager.cleanup()
    
    async def _initialize_tts_manager(self):
        """Initialisation du TTS Manager"""
        print("\n🔧 INITIALISATION TTS MANAGER")
        print("-" * 50)
        
        start_time = time.perf_counter()
        
        # Chargement de la configuration
        config_path = Path("config/tts.yaml")
        
        # Chargement du fichier YAML
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.tts_manager = UnifiedTTSManager(config)
        
        init_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ TTS Manager initialisé en {init_time:.1f}ms")
        print(f"📋 Backends configurés: piper_native_optimized, piper_native, piper_cli, sapi_french, silent_emergency")
    
    async def _test_text_court(self):
        """Test 1: Texte court (baseline)"""
        print("\n🔬 TEST 1: TEXTE COURT (BASELINE)")
        print("-" * 50)
        
        text = self.test_texts['court']
        print(f"📝 Texte: {len(text)} caractères")
        
        # Synthèse
        start_time = time.perf_counter()
        audio_data = await self.tts_manager.synthesize(text)
        synthesis_time = (time.perf_counter() - start_time) * 1000
        
        # Validation
        is_valid = is_valid_wav(audio_data)
        audio_info = get_wav_info(audio_data)
        
        print(f"✅ Synthèse réussie en {synthesis_time:.1f}ms")
        print(f"   Audio: {len(audio_data)} bytes, {audio_info.get('duration_ms', 0):.0f}ms")
        print(f"   Format WAV valide: {is_valid}")
        
        self.test_results['court'] = {
            'chars': len(text),
            'synthesis_time_ms': synthesis_time,
            'audio_size_bytes': len(audio_data),
            'audio_duration_ms': audio_info.get('duration_ms', 0),
            'audio_valid': is_valid
        }
    
    async def _test_text_moyen(self):
        """Test 2: Texte moyen (performance standard)"""
        print("\n🔬 TEST 2: TEXTE MOYEN (PERFORMANCE STANDARD)")
        print("-" * 50)
        
        text = self.test_texts['moyen'].strip()
        print(f"📝 Texte: {len(text)} caractères")
        
        # Synthèse
        start_time = time.perf_counter()
        audio_data = await self.tts_manager.synthesize(text)
        synthesis_time = (time.perf_counter() - start_time) * 1000
        
        # Validation
        is_valid = is_valid_wav(audio_data)
        audio_info = get_wav_info(audio_data)
        
        print(f"✅ Synthèse réussie en {synthesis_time:.1f}ms")
        print(f"   Audio: {len(audio_data)} bytes, {audio_info.get('duration_ms', 0):.0f}ms")
        print(f"   Format WAV valide: {is_valid}")
        
        # Calcul de la performance
        chars_per_ms = len(text) / synthesis_time if synthesis_time > 0 else 0
        print(f"🚀 Performance: {chars_per_ms:.2f} chars/ms")
        
        self.test_results['moyen'] = {
            'chars': len(text),
            'synthesis_time_ms': synthesis_time,
            'audio_size_bytes': len(audio_data),
            'audio_duration_ms': audio_info.get('duration_ms', 0),
            'audio_valid': is_valid,
            'chars_per_ms': chars_per_ms
        }
    
    async def _test_text_long(self):
        """Test 3: Texte long (chunking intelligent)"""
        print("\n🔬 TEST 3: TEXTE LONG (CHUNKING INTELLIGENT)")
        print("-" * 50)
        
        text = self.test_texts['long'].strip()
        print(f"📝 Texte: {len(text)} caractères")
        
        # Synthèse avec chunking
        start_time = time.perf_counter()
        audio_data = await self.tts_manager.synthesize(text)
        synthesis_time = (time.perf_counter() - start_time) * 1000
        
        # Validation
        is_valid = is_valid_wav(audio_data)
        audio_info = get_wav_info(audio_data)
        
        print(f"✅ Synthèse réussie en {synthesis_time:.1f}ms")
        print(f"   Audio: {len(audio_data)} bytes, {audio_info.get('duration_ms', 0):.0f}ms")
        print(f"   Format WAV valide: {is_valid}")
        
        # Estimation du nombre de chunks
        estimated_chunks = (len(text) // 800) + 1
        avg_time_per_chunk = synthesis_time / estimated_chunks
        print(f"📊 Chunks estimés: {estimated_chunks}")
        print(f"⚡ Temps moyen/chunk: {avg_time_per_chunk:.1f}ms")
        
        self.test_results['long'] = {
            'chars': len(text),
            'synthesis_time_ms': synthesis_time,
            'audio_size_bytes': len(audio_data),
            'audio_duration_ms': audio_info.get('duration_ms', 0),
            'audio_valid': is_valid,
            'estimated_chunks': estimated_chunks,
            'avg_time_per_chunk_ms': avg_time_per_chunk
        }
    
    async def _test_text_tres_long(self):
        """Test 4: Texte très long (5000+ chars)"""
        print("\n🔬 TEST 4: TEXTE TRÈS LONG (5000+ CHARS)")
        print("-" * 50)
        
        text = self.test_texts['tres_long'].strip()
        print(f"📝 Texte: {len(text)} caractères")
        
        # Synthèse avec chunking avancé
        start_time = time.perf_counter()
        audio_data = await self.tts_manager.synthesize(text)
        synthesis_time = (time.perf_counter() - start_time) * 1000
        
        # Validation
        is_valid = is_valid_wav(audio_data)
        audio_info = get_wav_info(audio_data)
        
        print(f"✅ Synthèse réussie en {synthesis_time:.1f}ms")
        print(f"   Audio: {len(audio_data)} bytes, {audio_info.get('duration_ms', 0):.0f}ms")
        print(f"   Format WAV valide: {is_valid}")
        
        # Métriques avancées
        estimated_chunks = (len(text) // 800) + 1
        avg_time_per_chunk = synthesis_time / estimated_chunks
        chars_per_second = (len(text) / synthesis_time) * 1000 if synthesis_time > 0 else 0
        
        print(f"📊 Chunks estimés: {estimated_chunks}")
        print(f"⚡ Temps moyen/chunk: {avg_time_per_chunk:.1f}ms")
        print(f"🚀 Vitesse traitement: {chars_per_second:.0f} chars/sec")
        
        # Vérification objectif Phase 3
        target_met = avg_time_per_chunk <= 100  # <100ms par chunk
        print(f"🎯 Objectif <100ms/chunk: {'✅' if target_met else '⚠️'}")
        
        self.test_results['tres_long'] = {
            'chars': len(text),
            'synthesis_time_ms': synthesis_time,
            'audio_size_bytes': len(audio_data),
            'audio_duration_ms': audio_info.get('duration_ms', 0),
            'audio_valid': is_valid,
            'estimated_chunks': estimated_chunks,
            'avg_time_per_chunk_ms': avg_time_per_chunk,
            'chars_per_second': chars_per_second,
            'target_met': target_met
        }
    
    async def _test_cache_performance(self):
        """Test 5: Performance du cache (texte récurrent)"""
        print("\n🔬 TEST 5: PERFORMANCE CACHE (TEXTE RÉCURRENT)")
        print("-" * 50)
        
        text = self.test_texts['recurrent']
        print(f"📝 Texte: {len(text)} caractères")
        
        # Premier appel (cache miss)
        print("🔄 Premier appel (cache miss)...")
        start_time = time.perf_counter()
        audio_data_1 = await self.tts_manager.synthesize(text)
        first_call_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   Temps: {first_call_time:.1f}ms")
        
        # Deuxième appel (cache hit attendu)
        print("⚡ Deuxième appel (cache hit attendu)...")
        start_time = time.perf_counter()
        audio_data_2 = await self.tts_manager.synthesize(text)
        second_call_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   Temps: {second_call_time:.1f}ms")
        
        # Analyse du cache
        cache_hit = second_call_time < (first_call_time * 0.1)  # 10x plus rapide
        speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
        
        print(f"🚀 Accélération: {speedup:.1f}x")
        print(f"💾 Cache hit détecté: {'✅' if cache_hit else '❌'}")
        
        # Validation identité audio
        audio_identical = audio_data_1 == audio_data_2
        print(f"🎵 Audio identique: {'✅' if audio_identical else '❌'}")
        
        self.test_results['cache'] = {
            'chars': len(text),
            'first_call_ms': first_call_time,
            'second_call_ms': second_call_time,
            'speedup': speedup,
            'cache_hit_detected': cache_hit,
            'audio_identical': audio_identical
        }
    
    def _generate_performance_report(self):
        """Génération du rapport de performance final"""
        print("\n" + "="*80)
        print("📊 RAPPORT PERFORMANCE PHASE 3 - CONDITIONS RÉELLES")
        print("="*80)
        
        # Résumé des tests
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if result.get('audio_valid', False))
        
        print(f"🎯 Tests réussis: {successful_tests}/{total_tests}")
        print()
        
        # Détail par test
        for test_name, result in self.test_results.items():
            print(f"📋 {test_name.upper()}:")
            print(f"   Caractères: {result['chars']}")
            print(f"   Temps synthèse: {result['synthesis_time_ms']:.1f}ms")
            
            if 'chars_per_ms' in result:
                print(f"   Performance: {result['chars_per_ms']:.2f} chars/ms")
            
            if 'avg_time_per_chunk_ms' in result:
                print(f"   Temps/chunk: {result['avg_time_per_chunk_ms']:.1f}ms")
            
            if 'speedup' in result:
                print(f"   Accélération cache: {result['speedup']:.1f}x")
            
            print(f"   Audio valide: {'✅' if result['audio_valid'] else '❌'}")
            print()
        
        # Métriques globales
        print("🚀 MÉTRIQUES GLOBALES:")
        
        # Performance moyenne
        synthesis_times = [r['synthesis_time_ms'] for r in self.test_results.values() 
                          if 'synthesis_time_ms' in r and r.get('audio_valid')]
        if synthesis_times:
            avg_time = sum(synthesis_times) / len(synthesis_times)
            print(f"   Temps moyen synthèse: {avg_time:.1f}ms")
        
        # Validation objectifs Phase 3
        print("\n🎯 OBJECTIFS PHASE 3:")
        
        # Objectif latence <100ms par chunk
        if 'tres_long' in self.test_results:
            target_met = self.test_results['tres_long'].get('target_met', False)
            print(f"   Latence <100ms/chunk: {'✅' if target_met else '⚠️'}")
        
        # Objectif textes longs (5000+ chars)
        long_text_supported = any(r['chars'] >= 5000 and r.get('audio_valid') 
                                 for r in self.test_results.values())
        print(f"   Support textes 5000+ chars: {'✅' if long_text_supported else '❌'}")
        
        # Objectif cache intelligent
        cache_working = self.test_results.get('cache', {}).get('cache_hit_detected', False)
        print(f"   Cache intelligent: {'✅' if cache_working else '❌'}")
        
        print("\n🎉 Tests de performance Phase 3 terminés!")


async def main():
    """Point d'entrée principal"""
    tester = Phase3PerformanceTester()
    await tester.run_performance_tests()


if __name__ == "__main__":
    asyncio.run(main()) 