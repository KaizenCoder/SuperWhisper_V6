import pytest
import numpy as np
import torch
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from STT.vad_manager import OptimizedVADManager

class TestOptimizedVADManager:
    """Tests unitaires pour STT/vad_manager.py avec coverage 85%"""
    
    @pytest.fixture
    def vad_manager(self):
        """Fixture VAD Manager avec configuration par défaut"""
        return OptimizedVADManager(chunk_ms=160, latency_threshold_ms=25)
    
    @pytest.fixture
    def custom_vad_manager(self):
        """Fixture VAD Manager avec configuration personnalisée"""
        return OptimizedVADManager(chunk_ms=320, latency_threshold_ms=15)
    
    @pytest.fixture
    def sample_audio_chunks(self):
        """Fixtures audio pour différents scénarios"""
        chunk_samples = int(16000 * 160 / 1000)  # 2560 échantillons
        
        return {
            'silence': np.zeros(chunk_samples, dtype=np.float32),
            'noise_low': np.random.randn(chunk_samples).astype(np.float32) * 0.01,
            'noise_medium': np.random.randn(chunk_samples).astype(np.float32) * 0.1,
            'noise_high': np.random.randn(chunk_samples).astype(np.float32) * 0.5,
            'speech_like': self._generate_speech_like_signal(chunk_samples),
            'too_short': np.random.randn(chunk_samples // 2).astype(np.float32),
            'too_long': np.random.randn(chunk_samples * 2).astype(np.float32)
        }
    
    def _generate_speech_like_signal(self, samples):
        """Génère un signal ressemblant à de la parole"""
        t = np.linspace(0, 1, samples)
        # Combinaison de fréquences typiques de la parole (300-3400 Hz)
        signal = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
            0.2 * np.sin(2 * np.pi * 880 * t) +  # A5  
            0.1 * np.sin(2 * np.pi * 1320 * t)   # E6
        )
        # Modulation d'amplitude pour simuler la parole
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
        return (signal * envelope).astype(np.float32)
    
    # Tests d'initialisation
    def test_init_default_params(self):
        """Test initialisation avec paramètres par défaut"""
        vad = OptimizedVADManager()
        
        assert vad.chunk_ms == 160
        assert vad.latency_threshold_ms == 25
        assert vad.chunk_samples == 2560  # 16000 * 160 / 1000
        assert vad.backend is None
        assert vad.vad_model is None
        assert vad.vad is None
    
    def test_init_custom_params(self, custom_vad_manager):
        """Test initialisation avec paramètres personnalisés"""
        assert custom_vad_manager.chunk_ms == 320
        assert custom_vad_manager.latency_threshold_ms == 15
        assert custom_vad_manager.chunk_samples == 5120  # 16000 * 320 / 1000
    
    # Tests d'initialisation async
    @pytest.mark.asyncio
    async def test_initialize_silero_fast_enough(self, vad_manager):
        """Test initialisation quand Silero est assez rapide"""
        with patch.object(vad_manager, '_test_silero_performance') as mock_silero, \
             patch.object(vad_manager, '_test_webrtc_performance') as mock_webrtc:
            
            mock_silero.return_value = 20.0  # < 25ms seuil
            
            await vad_manager.initialize()
            
            assert vad_manager.backend == "silero"
            mock_silero.assert_called_once()
            mock_webrtc.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_silero_too_slow_webrtc_ok(self, vad_manager):
        """Test initialisation quand Silero est lent mais WebRTC ok"""
        with patch.object(vad_manager, '_test_silero_performance') as mock_silero, \
             patch.object(vad_manager, '_test_webrtc_performance') as mock_webrtc:
            
            mock_silero.return_value = 30.0  # > 25ms seuil
            mock_webrtc.return_value = 5.0   # < 25ms seuil
            
            await vad_manager.initialize()
            
            assert vad_manager.backend == "webrtc"
            mock_silero.assert_called_once()
            mock_webrtc.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_both_too_slow(self, vad_manager):
        """Test initialisation quand tous les VAD sont trop lents"""
        with patch.object(vad_manager, '_test_silero_performance') as mock_silero, \
             patch.object(vad_manager, '_test_webrtc_performance') as mock_webrtc:
            
            mock_silero.return_value = 30.0  # > 25ms seuil
            mock_webrtc.return_value = 40.0  # > 25ms seuil
            
            await vad_manager.initialize()
            
            assert vad_manager.backend == "none"
            mock_silero.assert_called_once()
            mock_webrtc.assert_called_once()
    
    # Tests de performance Silero
    @pytest.mark.asyncio
    async def test_test_silero_performance_success(self, vad_manager):
        """Test mesure de performance Silero réussie"""
        mock_model = Mock()
        mock_utils = Mock()
        mock_model.return_value = torch.tensor(0.7)  # Probabilité de parole
        
        with patch('torch.hub.load') as mock_hub_load:
            mock_hub_load.return_value = (mock_model, mock_utils)
            
            latency = await vad_manager._test_silero_performance()
            
            assert isinstance(latency, float)
            assert latency >= 0
            assert vad_manager.vad_model == mock_model
            
            # Vérifier que le modèle a été appelé plusieurs fois (warmup + mesure)
            assert mock_model.call_count >= 25  # 5 warmup + 20 mesures
    
    @pytest.mark.asyncio
    async def test_test_silero_performance_failure(self, vad_manager):
        """Test gestion d'erreur lors du test Silero"""
        with patch('torch.hub.load') as mock_hub_load:
            mock_hub_load.side_effect = Exception("Model loading failed")
            
            latency = await vad_manager._test_silero_performance()
            
            assert latency == float('inf')
            assert vad_manager.vad_model is None
    
    # Tests de performance WebRTC
    @pytest.mark.asyncio
    async def test_test_webrtc_performance_success(self, vad_manager):
        """Test mesure de performance WebRTC réussie"""
        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        
        with patch('webrtcvad.Vad') as mock_webrtc_class:
            mock_webrtc_class.return_value = mock_vad
            
            latency = await vad_manager._test_webrtc_performance()
            
            assert isinstance(latency, float)
            assert latency >= 0
            assert vad_manager.vad == mock_vad
            
            # Vérifier que WebRTC a été appelé plusieurs fois (warmup + mesure)
            assert mock_vad.is_speech.call_count >= 60  # 10 warmup + 50 mesures
    
    @pytest.mark.asyncio
    async def test_test_webrtc_performance_import_error(self, vad_manager):
        """Test gestion d'erreur d'import WebRTC"""
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("webrtcvad not found")
            
            latency = await vad_manager._test_webrtc_performance()
            
            assert latency == float('inf')
            assert vad_manager.vad is None
    
    @pytest.mark.asyncio
    async def test_test_webrtc_performance_runtime_error(self, vad_manager):
        """Test gestion d'erreur runtime WebRTC"""
        with patch('webrtcvad.Vad') as mock_webrtc_class:
            mock_webrtc_class.side_effect = RuntimeError("WebRTC error")
            
            latency = await vad_manager._test_webrtc_performance()
            
            assert latency == float('inf')
    
    # Tests de détection de parole
    def test_is_speech_chunk_correct_size(self, vad_manager, sample_audio_chunks):
        """Test détection avec chunk de taille correcte"""
        vad_manager.backend = "none"  # Mode pass-through
        
        result = vad_manager.is_speech(sample_audio_chunks['noise_medium'])
        
        assert result is True  # Mode pass-through retourne toujours True
    
    def test_is_speech_chunk_too_short(self, vad_manager, sample_audio_chunks):
        """Test détection avec chunk trop court (padding)"""
        vad_manager.backend = "none"
        
        # Le chunk sera automatiquement paddé
        result = vad_manager.is_speech(sample_audio_chunks['too_short'])
        
        assert result is True
    
    def test_is_speech_chunk_too_long(self, vad_manager, sample_audio_chunks):
        """Test détection avec chunk trop long (troncature)"""
        vad_manager.backend = "none"
        
        # Le chunk sera automatiquement tronqué
        result = vad_manager.is_speech(sample_audio_chunks['too_long'])
        
        assert result is True
    
    def test_is_speech_silero_backend(self, vad_manager, sample_audio_chunks):
        """Test détection avec backend Silero"""
        # Configuration du backend Silero
        vad_manager.backend = "silero"
        mock_model = Mock()
        mock_model.return_value = torch.tensor(0.8)  # Probabilité élevée
        vad_manager.vad_model = mock_model
        
        with patch('torch.no_grad'):
            result = vad_manager.is_speech(sample_audio_chunks['speech_like'])
        
        assert result is True  # 0.8 > 0.5 seuil
        mock_model.assert_called_once()
    
    def test_is_speech_silero_low_probability(self, vad_manager, sample_audio_chunks):
        """Test détection Silero avec probabilité faible"""
        vad_manager.backend = "silero"
        mock_model = Mock()
        mock_model.return_value = torch.tensor(0.3)  # Probabilité faible
        vad_manager.vad_model = mock_model
        
        with patch('torch.no_grad'):
            result = vad_manager.is_speech(sample_audio_chunks['silence'])
        
        assert result is False  # 0.3 < 0.5 seuil
    
    def test_is_speech_silero_error_fallback(self, vad_manager, sample_audio_chunks):
        """Test fallback Silero en cas d'erreur"""
        vad_manager.backend = "silero"
        mock_model = Mock()
        mock_model.side_effect = RuntimeError("Silero error")
        vad_manager.vad_model = mock_model
        
        with patch('torch.no_grad'):
            result = vad_manager.is_speech(sample_audio_chunks['noise_medium'])
        
        assert result is True  # Fallback: considérer comme parole
    
    def test_is_speech_webrtc_backend(self, vad_manager, sample_audio_chunks):
        """Test détection avec backend WebRTC"""
        vad_manager.backend = "webrtc"
        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        vad_manager.vad = mock_vad
        
        result = vad_manager.is_speech(sample_audio_chunks['speech_like'])
        
        assert result is True
        mock_vad.is_speech.assert_called_once()
    
    def test_is_speech_webrtc_error_fallback(self, vad_manager, sample_audio_chunks):
        """Test fallback WebRTC en cas d'erreur"""
        vad_manager.backend = "webrtc"
        mock_vad = Mock()
        mock_vad.is_speech.side_effect = Exception("WebRTC error")
        vad_manager.vad = mock_vad
        
        result = vad_manager.is_speech(sample_audio_chunks['noise_medium'])
        
        assert result is True  # Fallback: considérer comme parole
    
    # Tests de probabilité de parole
    def test_get_speech_probability_silero(self, vad_manager, sample_audio_chunks):
        """Test probabilité de parole avec Silero"""
        vad_manager.backend = "silero"
        mock_model = Mock()
        mock_model.return_value = torch.tensor(0.75)
        vad_manager.vad_model = mock_model
        
        with patch('torch.no_grad'):
            prob = vad_manager.get_speech_probability(sample_audio_chunks['speech_like'])
        
        assert prob == 0.75
    
    def test_get_speech_probability_silero_error(self, vad_manager, sample_audio_chunks):
        """Test probabilité Silero avec erreur"""
        vad_manager.backend = "silero"
        mock_model = Mock()
        mock_model.side_effect = Exception("Error")
        vad_manager.vad_model = mock_model
        
        with patch('torch.no_grad'):
            prob = vad_manager.get_speech_probability(sample_audio_chunks['noise_medium'])
        
        assert prob == 0.5  # Fallback neutre
    
    def test_get_speech_probability_webrtc_true(self, vad_manager, sample_audio_chunks):
        """Test probabilité WebRTC (parole détectée)"""
        vad_manager.backend = "webrtc"
        
        with patch.object(vad_manager, 'is_speech', return_value=True):
            prob = vad_manager.get_speech_probability(sample_audio_chunks['speech_like'])
        
        assert prob == 1.0
    
    def test_get_speech_probability_webrtc_false(self, vad_manager, sample_audio_chunks):
        """Test probabilité WebRTC (pas de parole)"""
        vad_manager.backend = "webrtc"
        
        with patch.object(vad_manager, 'is_speech', return_value=False):
            prob = vad_manager.get_speech_probability(sample_audio_chunks['silence'])
        
        assert prob == 0.0
    
    def test_get_speech_probability_none_backend(self, vad_manager, sample_audio_chunks):
        """Test probabilité avec backend none"""
        vad_manager.backend = "none"
        
        prob = vad_manager.get_speech_probability(sample_audio_chunks['noise_medium'])
        
        assert prob == 1.0  # Mode pass-through
    
    # Tests de benchmark de performance
    def test_benchmark_performance_none_backend(self, vad_manager):
        """Test benchmark avec backend none"""
        vad_manager.backend = "none"
        
        stats = vad_manager.benchmark_performance(10)
        
        assert stats["backend"] == "none"
        assert stats["avg_latency_ms"] == 0.0
    
    def test_benchmark_performance_with_backend(self, vad_manager):
        """Test benchmark avec backend actif"""
        vad_manager.backend = "silero"
        
        # Mock de is_speech pour éviter les dépendances
        with patch.object(vad_manager, 'is_speech', return_value=True):
            stats = vad_manager.benchmark_performance(5)
        
        assert stats["backend"] == "silero"
        assert "avg_latency_ms" in stats
        assert "std_latency_ms" in stats
        assert "max_latency_ms" in stats
        assert "min_latency_ms" in stats
        assert stats["chunk_ms"] == 160
        assert stats["iterations"] == 5
        assert isinstance(stats["avg_latency_ms"], float)
        assert stats["avg_latency_ms"] >= 0
    
    @pytest.mark.performance
    def test_benchmark_performance_sla_compliance(self, vad_manager, sample_audio_chunks):
        """Test que le benchmark respecte le SLA de <25ms"""
        vad_manager.backend = "silero"
        
        # Mock optimisé pour simuler performance rapide
        def fast_is_speech(chunk):
            time.sleep(0.001)  # Simule 1ms de traitement
            return True
        
        with patch.object(vad_manager, 'is_speech', side_effect=fast_is_speech):
            stats = vad_manager.benchmark_performance(10)
        
        # Vérification SLA: latence moyenne < 25ms
        assert stats["avg_latency_ms"] < 25.0, f"SLA violé: {stats['avg_latency_ms']:.2f}ms >= 25ms"
        assert stats["max_latency_ms"] < 50.0, f"Latence max trop élevée: {stats['max_latency_ms']:.2f}ms"
    
    # Tests de statut
    def test_get_status_initialized(self, vad_manager):
        """Test statut avec VAD initialisé"""
        vad_manager.backend = "silero"
        
        status = vad_manager.get_status()
        
        expected_status = {
            "backend": "silero",
            "chunk_ms": 160,
            "chunk_samples": 2560,
            "latency_threshold_ms": 25,
            "initialized": True
        }
        assert status == expected_status
    
    def test_get_status_not_initialized(self, vad_manager):
        """Test statut avec VAD non initialisé"""
        status = vad_manager.get_status()
        
        expected_status = {
            "backend": None,
            "chunk_ms": 160,
            "chunk_samples": 2560,
            "latency_threshold_ms": 25,
            "initialized": False
        }
        assert status == expected_status
    
    # Tests de performance et gestion de taille
    def test_chunk_size_calculation(self):
        """Test calcul correct de la taille des chunks"""
        # Test différentes durées
        test_cases = [
            (160, 2560),   # 160ms -> 2560 échantillons
            (320, 5120),   # 320ms -> 5120 échantillons
            (80, 1280),    # 80ms -> 1280 échantillons
            (240, 3840)    # 240ms -> 3840 échantillons
        ]
        
        for chunk_ms, expected_samples in test_cases:
            vad = OptimizedVADManager(chunk_ms=chunk_ms)
            assert vad.chunk_samples == expected_samples
    
    def test_audio_chunk_padding(self, vad_manager):
        """Test padding automatique des chunks trop courts"""
        vad_manager.backend = "none"
        
        # Chunk moitié taille
        short_chunk = np.random.randn(vad_manager.chunk_samples // 2).astype(np.float32)
        
        # Le traitement doit fonctionner sans erreur (padding automatique)
        result = vad_manager.is_speech(short_chunk)
        assert isinstance(result, bool)
    
    def test_audio_chunk_truncation(self, vad_manager):
        """Test troncature automatique des chunks trop longs"""
        vad_manager.backend = "none"
        
        # Chunk double taille
        long_chunk = np.random.randn(vad_manager.chunk_samples * 2).astype(np.float32)
        
        # Le traitement doit fonctionner sans erreur (troncature automatique)
        result = vad_manager.is_speech(long_chunk)
        assert isinstance(result, bool)
    
    # Tests d'intégration et cas limites
    def test_empty_audio_chunk(self, vad_manager):
        """Test avec chunk audio vide"""
        vad_manager.backend = "none"
        
        empty_chunk = np.array([], dtype=np.float32)
        
        # Doit fonctionner avec padding automatique
        result = vad_manager.is_speech(empty_chunk)
        assert result is True  # Mode pass-through
    
    def test_extreme_audio_values(self, vad_manager):
        """Test avec valeurs audio extrêmes"""
        vad_manager.backend = "none"
        
        # Valeurs très élevées
        extreme_high = np.full(vad_manager.chunk_samples, 10.0, dtype=np.float32)
        result_high = vad_manager.is_speech(extreme_high)
        assert isinstance(result_high, bool)
        
        # Valeurs très faibles
        extreme_low = np.full(vad_manager.chunk_samples, -10.0, dtype=np.float32)
        result_low = vad_manager.is_speech(extreme_low)
        assert isinstance(result_low, bool)
    
    def test_nan_audio_values(self, vad_manager):
        """Test avec valeurs NaN dans l'audio"""
        vad_manager.backend = "none"
        
        nan_chunk = np.full(vad_manager.chunk_samples, np.nan, dtype=np.float32)
        
        # Doit gérer gracieusement les NaN
        result = vad_manager.is_speech(nan_chunk)
        assert isinstance(result, bool)
    
    def test_inf_audio_values(self, vad_manager):
        """Test avec valeurs infinies dans l'audio"""
        vad_manager.backend = "none"
        
        inf_chunk = np.full(vad_manager.chunk_samples, np.inf, dtype=np.float32)
        
        # Doit gérer gracieusement les infinis
        result = vad_manager.is_speech(inf_chunk)
        assert isinstance(result, bool)
    
    # Tests de concurrence
    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Test initialisation concurrente de plusieurs VAD"""
        async def init_vad(chunk_ms):
            vad = OptimizedVADManager(chunk_ms=chunk_ms, latency_threshold_ms=50)
            with patch.object(vad, '_test_silero_performance', return_value=20.0), \
                 patch.object(vad, '_test_webrtc_performance', return_value=10.0):
                await vad.initialize()
                return vad.backend
        
        # Initialiser 3 VAD en parallèle
        tasks = [init_vad(160), init_vad(320), init_vad(80)]
        backends = await asyncio.gather(*tasks)
        
        # Tous doivent avoir sélectionné silero (latence 20ms < 50ms seuil)
        assert all(backend == "silero" for backend in backends)
    
    def test_memory_usage_monitoring(self, vad_manager, sample_audio_chunks):
        """Test surveillance utilisation mémoire"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        vad_manager.backend = "none"
        
        # Traiter plusieurs chunks
        for _ in range(100):
            vad_manager.is_speech(sample_audio_chunks['noise_medium'])
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Vérification: pas de fuite mémoire majeure
        assert memory_increase < 50, f"Augmentation mémoire excessive: {memory_increase:.1f}MB"


class TestVADManagerIntegration:
    """Tests d'intégration VAD Manager"""
    
    @pytest.mark.asyncio
    async def test_full_vad_workflow(self):
        """Test workflow complet VAD"""
        vad = OptimizedVADManager(chunk_ms=160, latency_threshold_ms=30)
        
        # Mock des backends pour test reproductible
        with patch.object(vad, '_test_silero_performance', return_value=25.0), \
             patch.object(vad, '_test_webrtc_performance', return_value=35.0):
            
            # Initialisation
            await vad.initialize()
            assert vad.backend == "silero"  # 25ms < 30ms seuil
            
            # Test détection
            chunk = np.random.randn(vad.chunk_samples).astype(np.float32)
            
            with patch.object(vad, '_is_speech_silero', return_value=True):
                is_speech = vad.is_speech(chunk)
                probability = vad.get_speech_probability(chunk)
                
                assert is_speech is True
                assert isinstance(probability, float)
                assert 0.0 <= probability <= 1.0
            
            # Benchmark
            with patch.object(vad, 'is_speech', return_value=True):
                stats = vad.benchmark_performance(5)
                assert stats["backend"] == "silero"
                assert stats["avg_latency_ms"] >= 0
            
            # Statut final
            status = vad.get_status()
            assert status["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_fallback_cascade(self):
        """Test cascade de fallback: Silero -> WebRTC -> None"""
        vad = OptimizedVADManager(chunk_ms=160, latency_threshold_ms=20)
        
        # Test: Silero trop lent, WebRTC ok
        with patch.object(vad, '_test_silero_performance', return_value=30.0), \
             patch.object(vad, '_test_webrtc_performance', return_value=15.0):
            
            await vad.initialize()
            assert vad.backend == "webrtc"
        
        # Reset pour test suivant
        vad = OptimizedVADManager(chunk_ms=160, latency_threshold_ms=10)
        
        # Test: Tous trop lents -> mode none
        with patch.object(vad, '_test_silero_performance', return_value=30.0), \
             patch.object(vad, '_test_webrtc_performance', return_value=25.0):
            
            await vad.initialize()
            assert vad.backend == "none"


if __name__ == "__main__":
    # Exécution des tests avec coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=STT.vad_manager",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=85"
    ]) 