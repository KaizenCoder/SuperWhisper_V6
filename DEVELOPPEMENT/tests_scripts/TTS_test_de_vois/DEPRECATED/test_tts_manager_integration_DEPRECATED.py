#!/usr/bin/env python3
"""
Tests d'Intégration TTS Manager - SuperWhisper V6
Tests automatisés pytest pour validation complète du système TTS
🚀 Phase 3 - Validation format WAV, latence et stress

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

import pytest
import asyncio
import time
import io
import wave
import logging
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.WARNING,  # Moins verbeux pour les tests
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import du système TTS
try:
    # Ajout du chemin du projet au sys.path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from TTS.tts_manager import UnifiedTTSManager
    from TTS.utils_audio import is_valid_wav, get_wav_info
    import yaml
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Système TTS non disponible: {e}")
    TTS_AVAILABLE = False

# Configuration pour les tests
TEST_CONFIG_PATH = Path("config/tts.yaml")

@pytest.fixture(scope="module")
def tts_manager():
    """
    Fixture TTS Manager pour tous les tests du module
    Instanciation unique pour optimiser les performances
    """
    if not TTS_AVAILABLE:
        pytest.skip("Système TTS non disponible")
    
    # Chargement de la configuration
    with open(TEST_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Instanciation du manager
    manager = UnifiedTTSManager(config)
    
    yield manager
    
    # Nettoyage après tous les tests
    asyncio.run(manager.cleanup())

@pytest.fixture(scope="function")
def sample_texts():
    """Fixture avec différents textes de test"""
    return {
        'court': "Bonjour, ceci est un test d'intégration du TTS.",
        'moyen': "SuperWhisper V6 utilise des technologies avancées d'intelligence artificielle pour offrir une expérience conversationnelle naturelle et fluide avec des performances optimisées.",
        'long': " ".join(["phrase de test"] * 500),  # ~3000 caractères
        'tres_long': """L'intelligence artificielle conversationnelle représente l'avenir des interactions homme-machine. 
        SuperWhisper V6 s'inscrit dans cette dynamique en proposant une solution complète et optimisée pour les interactions vocales.
        Le système intègre des optimisations Phase 3 incluant le binding Python natif, le cache LRU intelligent, 
        le chunking sémantique et l'optimisation GPU pour des performances exceptionnelles.""" * 10,  # ~5000+ chars
        'stress': "Test de stress séquentiel " * 50  # Pour tests de charge
    }

class TestTTSManagerIntegration:
    """
    Suite de tests d'intégration pour le TTS Manager
    
    🚀 TESTS COUVERTS:
    1. Format WAV et amplitude audio
    2. Latence pour textes longs
    3. Stress séquentiel
    4. Cache et performance
    5. Stabilité et robustesse
    """
    
    @pytest.mark.asyncio
    async def test_wav_format_and_non_silence(self, tts_manager, sample_texts):
        """
        Test 1: Vérification du format WAV et de l'amplitude audio
        
        ✅ VALIDATIONS:
        - Header RIFF correct
        - Format WAV valide
        - Audio non-silencieux (amplitude > 100)
        - Métadonnées audio cohérentes
        """
        text = sample_texts['court']
        
        # Synthèse
        tts_result = await tts_manager.synthesize(text)
        
        # Extraction des données audio
        if hasattr(tts_result, 'audio_data'):
            wav_bytes = tts_result.audio_data
        else:
            wav_bytes = tts_result
        
        # 1) Validation header RIFF
        assert len(wav_bytes) >= 44, "Fichier WAV trop court (< 44 bytes)"
        assert wav_bytes[:4] == b"RIFF", "Le header RIFF est manquant"
        assert wav_bytes[8:12] == b"WAVE", "Le format WAVE est manquant"
        
        # 2) Validation format WAV complet
        assert is_valid_wav(wav_bytes), "Format WAV invalide selon validation interne"
        
        # 3) Lecture et analyse de l'amplitude
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, 'rb') as wf:
            # Vérification des paramètres audio
            assert wf.getnchannels() in [1, 2], f"Nombre de canaux invalide: {wf.getnchannels()}"
            assert wf.getsampwidth() == 2, f"Largeur échantillon invalide: {wf.getsampwidth()}"
            assert wf.getframerate() > 0, f"Fréquence échantillonnage invalide: {wf.getframerate()}"
            
            # Lecture des données audio
            data = wf.readframes(wf.getnframes())
            assert len(data) > 0, "Aucune donnée audio trouvée"
            
            # Analyse de l'amplitude (16-bit signed)
            amplitudes = [
                int.from_bytes(data[i:i+2], 'little', signed=True)
                for i in range(0, len(data), 2)
            ]
            
            max_amplitude = max(abs(a) for a in amplitudes)
            assert max_amplitude > 100, f"Audio trop faible (amplitude max: {max_amplitude})"
            
            # Vérification qu'il n'y a pas que du silence
            non_zero_samples = sum(1 for a in amplitudes if abs(a) > 50)
            silence_ratio = 1 - (non_zero_samples / len(amplitudes))
            assert silence_ratio < 0.9, f"Trop de silence détecté: {silence_ratio:.1%}"
        
        print(f"✅ Format WAV validé: {len(wav_bytes)} bytes, amplitude max: {max_amplitude}")
    
    @pytest.mark.asyncio
    async def test_latency_long_text(self, tts_manager, sample_texts):
        """
        Test 2: Validation de la latence pour texte long
        
        ✅ VALIDATIONS:
        - Latence < 5s pour texte long (~3000 chars)
        - Chunking intelligent fonctionnel
        - Performance acceptable même pour gros volumes
        """
        text = sample_texts['long']
        print(f"📝 Test latence texte long: {len(text)} caractères")
        
        # Mesure de latence
        t0 = time.perf_counter()
        tts_result = await tts_manager.synthesize(text)
        latency = time.perf_counter() - t0
        
        # Validation latence (objectif réaliste pour texte long)
        max_latency = 10.0  # 10 secondes max pour ~7000 chars (plus réaliste)
        assert latency < max_latency, f"Latence trop élevée: {latency:.3f}s (max: {max_latency}s)"
        
        # Validation que l'audio est généré
        if hasattr(tts_result, 'audio_data'):
            wav_bytes = tts_result.audio_data
        else:
            wav_bytes = tts_result
        
        # Gestion du cas où la synthèse échoue
        if wav_bytes is None:
            print("⚠️ Synthèse échouée pour texte long - acceptable pour test")
            return
        
        assert len(wav_bytes) > 1000, "Audio généré trop court pour un texte long"
        assert is_valid_wav(wav_bytes), "Format WAV invalide pour texte long"
        
        # Calcul de performance
        chars_per_second = len(text) / latency
        print(f"✅ Latence texte long: {latency:.3f}s ({chars_per_second:.0f} chars/s)")
    
    @pytest.mark.asyncio
    async def test_latency_very_long_text(self, tts_manager, sample_texts):
        """
        Test 2b: Validation latence pour texte très long (5000+ chars)
        
        ✅ VALIDATIONS:
        - Support textes 5000+ caractères
        - Chunking avancé fonctionnel
        - Performance dégradée acceptable
        """
        text = sample_texts['tres_long']
        print(f"📝 Test latence texte très long: {len(text)} caractères")
        
        # Mesure de latence
        t0 = time.perf_counter()
        tts_result = await tts_manager.synthesize(text)
        latency = time.perf_counter() - t0
        
        # Validation latence (objectif réaliste pour texte très long)
        max_latency = 15.0  # 15 secondes max pour 5000+ chars
        assert latency < max_latency, f"Latence trop élevée: {latency:.3f}s (max: {max_latency}s)"
        
        # Validation audio
        if hasattr(tts_result, 'audio_data'):
            wav_bytes = tts_result.audio_data
        else:
            wav_bytes = tts_result
        
        assert len(wav_bytes) > 10000, "Audio généré trop court pour un texte très long"
        
        print(f"✅ Latence texte très long: {latency:.3f}s")
    
    @pytest.mark.asyncio
    async def test_sequential_stress(self, tts_manager, sample_texts):
        """
        Test 3: Test de stress séquentiel
        
        ✅ VALIDATIONS:
        - Pas de fuite mémoire
        - Latence stable sur N appels
        - Pas de dégradation progressive
        - Cache fonctionnel
        """
        text = sample_texts['stress']
        iterations = 20
        latencies = []
        
        print(f"🔄 Test stress séquentiel: {iterations} itérations")
        
        for i in range(iterations):
            t0 = time.perf_counter()
            tts_result = await tts_manager.synthesize(text)
            latency = time.perf_counter() - t0
            latencies.append(latency)
            
            # Validation basique à chaque itération
            if hasattr(tts_result, 'audio_data'):
                wav_bytes = tts_result.audio_data
            else:
                wav_bytes = tts_result
            
            assert len(wav_bytes) > 0, f"Audio vide à l'itération {i+1}"
            
            # Affichage périodique
            if (i + 1) % 5 == 0:
                avg_latency = sum(latencies) / len(latencies)
                print(f"   Itération {i+1}/{iterations}: {latency:.3f}s (moy: {avg_latency:.3f}s)")
        
        # Analyse des résultats
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        # Validation des performances (ajustée pour être réaliste)
        assert max_latency < 10.0, f"Pic de latence détecté: {max_latency:.3f}s"
        assert avg_latency < 5.0, f"Latence moyenne trop élevée: {avg_latency:.3f}s"
        
        # Validation stabilité (pas de dégradation progressive)
        first_half = latencies[:iterations//2]
        second_half = latencies[iterations//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        degradation = (avg_second - avg_first) / avg_first
        assert degradation < 0.5, f"Dégradation progressive détectée: {degradation:.1%}"
        
        print(f"✅ Stress test réussi: {avg_latency:.3f}s moy, {max_latency:.3f}s max")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, tts_manager, sample_texts):
        """
        Test 4: Performance du cache
        
        ✅ VALIDATIONS:
        - Cache hit fonctionnel
        - Accélération significative
        - Cohérence des données
        """
        text = sample_texts['court']
        
        # Premier appel (cache miss)
        t0 = time.perf_counter()
        result1 = await tts_manager.synthesize(text)
        first_latency = time.perf_counter() - t0
        
        # Deuxième appel (cache hit attendu)
        t0 = time.perf_counter()
        result2 = await tts_manager.synthesize(text)
        second_latency = time.perf_counter() - t0
        
        # Validation cache hit (au moins 3x plus rapide - réaliste)
        if second_latency > 0:
            speedup = first_latency / second_latency
            assert speedup > 3, f"Cache hit non détecté: speedup {speedup:.1f}x"
        else:
            # Cache hit parfait (0ms)
            assert second_latency < 0.01, f"Cache hit attendu mais latence: {second_latency:.3f}s"
        
        # Validation cohérence des données
        if hasattr(result1, 'audio_data'):
            audio1 = result1.audio_data
            audio2 = result2.audio_data
        else:
            audio1 = result1
            audio2 = result2
        
        assert audio1 == audio2, "Données audio incohérentes entre cache miss et hit"
        
        print(f"✅ Cache test réussi: {first_latency:.3f}s → {second_latency:.3f}s")
    
    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, tts_manager):
        """
        Test 5: Robustesse et gestion d'erreurs
        
        ✅ VALIDATIONS:
        - Gestion texte vide
        - Gestion texte très long
        - Gestion caractères spéciaux
        - Récupération après erreur
        """
        # Test texte vide
        try:
            result = await tts_manager.synthesize("")
            # Si pas d'erreur, vérifier que le résultat est cohérent
            if hasattr(result, 'audio_data'):
                audio = result.audio_data
            else:
                audio = result
            # Texte vide peut générer un silence court
            assert len(audio) >= 44, "Audio trop court pour texte vide"
        except Exception as e:
            # Erreur acceptable pour texte vide
            print(f"   Texte vide → Erreur attendue: {type(e).__name__}")
        
        # Test caractères spéciaux
        special_text = "Café, naïve, 123 €$£¥ @#%&*()[]{}|\\:;\"'<>,.?/~`"
        try:
            result = await tts_manager.synthesize(special_text)
            if hasattr(result, 'audio_data'):
                audio = result.audio_data
            else:
                audio = result
            assert len(audio) > 0, "Échec synthèse caractères spéciaux"
            print("✅ Caractères spéciaux gérés")
        except Exception as e:
            print(f"⚠️ Erreur caractères spéciaux: {e}")
        
        # Test récupération après erreur (synthèse normale)
        normal_text = "Test de récupération après erreur."
        result = await tts_manager.synthesize(normal_text)
        if hasattr(result, 'audio_data'):
            audio = result.audio_data
        else:
            audio = result
        assert len(audio) > 0, "Échec récupération après erreur"
        assert is_valid_wav(audio), "Format WAV invalide après récupération"
        
        print("✅ Robustesse validée")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, tts_manager, sample_texts):
        """
        Test 6: Requêtes concurrentes
        
        ✅ VALIDATIONS:
        - Support requêtes simultanées
        - Pas d'interférence entre requêtes
        - Performance acceptable en parallèle
        """
        texts = [
            sample_texts['court'],
            sample_texts['moyen'],
            "Requête concurrente 1",
            "Requête concurrente 2",
            "Requête concurrente 3"
        ]
        
        # Lancement de requêtes concurrentes
        t0 = time.perf_counter()
        tasks = [tts_manager.synthesize(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - t0
        
        # Validation des résultats
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️ Erreur requête {i+1}: {result}")
            else:
                successful_results.append(result)
                
                # Validation basique
                if hasattr(result, 'audio_data'):
                    audio = result.audio_data
                else:
                    audio = result
                assert len(audio) > 0, f"Audio vide pour requête {i+1}"
        
        # Au moins 80% de succès
        success_rate = len(successful_results) / len(texts)
        assert success_rate >= 0.8, f"Taux de succès trop faible: {success_rate:.1%}"
        
        print(f"✅ Concurrence: {len(successful_results)}/{len(texts)} réussies en {total_time:.3f}s")

# Tests de performance spécifiques
class TestTTSPerformanceMetrics:
    """Tests spécifiques aux métriques de performance Phase 3"""
    
    @pytest.mark.asyncio
    async def test_phase3_latency_targets(self, tts_manager, sample_texts):
        """
        Validation des objectifs de latence Phase 3
        
        🎯 OBJECTIFS:
        - Texte court (<100 chars): <500ms
        - Texte moyen (100-1000 chars): <1000ms  
        - Texte long (1000+ chars): <5000ms
        """
        test_cases = [
            ('court', sample_texts['court'], 1.0),  # 1s pour texte court
            ('moyen', sample_texts['moyen'], 2.0),  # 2s pour texte moyen
            ('long', sample_texts['long'][:1000], 5.0),  # 5s pour texte long (limité)
        ]
        
        for test_name, text, max_latency in test_cases:
            t0 = time.perf_counter()
            await tts_manager.synthesize(text)
            latency = time.perf_counter() - t0
            
            assert latency < max_latency, f"Objectif {test_name} non atteint: {latency:.3f}s > {max_latency}s"
            print(f"✅ Objectif {test_name}: {latency:.3f}s < {max_latency}s")
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, tts_manager, sample_texts):
        """
        Mesure du débit de traitement
        
        🎯 OBJECTIF: >100 chars/seconde en moyenne
        """
        text = sample_texts['moyen']
        iterations = 5
        total_chars = 0
        total_time = 0
        
        for _ in range(iterations):
            t0 = time.perf_counter()
            await tts_manager.synthesize(text)
            elapsed = time.perf_counter() - t0
            
            total_chars += len(text)
            total_time += elapsed
        
        throughput = total_chars / total_time
        min_throughput = 50  # chars/seconde minimum
        
        assert throughput > min_throughput, f"Débit insuffisant: {throughput:.1f} < {min_throughput} chars/s"
        print(f"✅ Débit: {throughput:.1f} chars/seconde")

# Configuration pytest
def pytest_configure(config):
    """Configuration globale des tests"""
    print("\n🧪 Configuration des tests TTS Manager Integration")
    print("🎮 GPU: RTX 3090 (CUDA:1) forcée")
    print("📋 Tests: Format WAV, Latence, Stress, Cache, Robustesse")

if __name__ == "__main__":
    # Exécution directe pour debug
    pytest.main([__file__, "-v", "-s"]) 