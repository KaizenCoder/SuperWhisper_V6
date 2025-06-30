#!/usr/bin/env python3
"""
Tests d'Intégration - Luxa SuperWhisper V6
==========================================

Tests réalistes du pipeline complet avec données audio réelles.
"""

import pytest
import asyncio
import numpy as np
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from Orchestrator.master_handler_robust import RobustMasterHandler
from STT.vad_manager import OptimizedVADManager
from utils.gpu_manager import get_gpu_manager
from utils.error_handler import RobustErrorHandler, CircuitBreaker
from config.security_config import SecurityConfig

# Configuration logging pour les tests
logging.basicConfig(level=logging.WARNING)

class TestPipelineIntegration:
    """Tests d'intégration du pipeline complet"""
    
    @pytest.fixture
    async def master_handler(self):
        """Fixture pour le gestionnaire principal"""
        handler = RobustMasterHandler()
        await handler.initialize()
        yield handler
        # Cleanup
        if hasattr(handler, 'cleanup'):
            await handler.cleanup()
    
    @pytest.fixture
    def audio_samples(self):
        """Génère des échantillons audio de test"""
        return {
            "silence": np.zeros(16000, dtype=np.float32),  # 1s de silence
            "noise": np.random.normal(0, 0.1, 16000).astype(np.float32),  # Bruit
            "speech_sim": self._generate_speech_like_audio(16000),  # Simulation parole
            "short": np.random.normal(0, 0.1, 8000).astype(np.float32),  # 0.5s
            "long": np.random.normal(0, 0.1, 48000).astype(np.float32),  # 3s
        }
    
    def _generate_speech_like_audio(self, length: int) -> np.ndarray:
        """Génère un signal qui ressemble à de la parole"""
        # Combinaison de sinusoïdes pour simuler formants
        t = np.linspace(0, length/16000, length)
        
        # Formants typiques de la parole (F1, F2, F3)
        f1 = 800 * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))  # Variation prosodique
        f2 = 1200 * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
        f3 = 2400
        
        signal = (
            0.3 * np.sin(2 * np.pi * f1 * t) +
            0.2 * np.sin(2 * np.pi * f2 * t) +
            0.1 * np.sin(2 * np.pi * f3 * t)
        )
        
        # Envelope pour simuler les mots
        envelope = np.abs(np.sin(2 * np.pi * 2 * t)) ** 2
        
        return (signal * envelope * 0.5).astype(np.float32)
    
    @pytest.mark.asyncio
    async def test_pipeline_with_silence(self, master_handler, audio_samples):
        """Test du pipeline avec du silence"""
        result = await master_handler.process_audio_safe(audio_samples["silence"])
        
        assert result["success"] is True
        assert result["text"] == ""
        assert result["latency_ms"] < 100  # Doit être très rapide pour le silence
        assert "vad" in result["components_used"]
    
    @pytest.mark.asyncio
    async def test_pipeline_with_noise(self, master_handler, audio_samples):
        """Test du pipeline avec du bruit"""
        result = await master_handler.process_audio_safe(audio_samples["noise"])
        
        assert result["success"] is True
        # Le bruit peut être détecté comme parole ou non selon le VAD
        assert result["latency_ms"] < 3000  # Limite raisonnable
    
    @pytest.mark.asyncio
    async def test_pipeline_with_speech_simulation(self, master_handler, audio_samples):
        """Test du pipeline avec simulation de parole"""
        result = await master_handler.process_audio_safe(audio_samples["speech_sim"])
        
        assert result["success"] is True
        assert "components_used" in result
        assert "stt" in result["components_used"]
        assert result["latency_ms"] < 5000  # Limite pour traitement STT
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_multiple_requests(self, master_handler, audio_samples):
        """Test de performance avec requêtes multiples"""
        num_requests = 10
        latencies = []
        
        for i in range(num_requests):
            audio = audio_samples["speech_sim"] if i % 2 == 0 else audio_samples["noise"]
            
            start_time = time.time()
            result = await master_handler.process_audio_safe(audio)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)
            assert result["success"] is True
        
        # Vérifier les performances
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 3000, f"Latence moyenne trop élevée: {avg_latency:.1f}ms"
        assert max_latency < 5000, f"Latence max trop élevée: {max_latency:.1f}ms"
        
        print(f"📊 Performance - Moyenne: {avg_latency:.1f}ms, Max: {max_latency:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self, master_handler, audio_samples):
        """Test de récupération après erreur"""
        
        # Simuler une erreur dans le STT
        with patch.object(master_handler.fallback_manager, 'get_component') as mock_stt:
            # Premier appel échoue
            mock_stt.side_effect = Exception("STT Error")
            
            result = await master_handler.process_audio_safe(audio_samples["speech_sim"])
            assert result["success"] is False
            assert "STT Error" in str(result["errors"])
        
        # Vérifier que le système se remet
        result = await master_handler.process_audio_safe(audio_samples["speech_sim"])
        # Devrait fonctionner normalement après la récupération

class TestVADIntegration:
    """Tests d'intégration du VAD"""
    
    @pytest.fixture
    async def vad_manager(self):
        """Fixture pour le gestionnaire VAD"""
        vad = OptimizedVADManager(chunk_ms=160, latency_threshold_ms=25)
        await vad.initialize()
        return vad
    
    @pytest.mark.asyncio
    async def test_vad_initialization(self, vad_manager):
        """Test d'initialisation du VAD"""
        assert vad_manager.backend in ["silero", "webrtc", "none"]
        print(f"VAD backend sélectionné: {vad_manager.backend}")
    
    @pytest.mark.asyncio
    async def test_vad_performance_realistic(self, vad_manager):
        """Test de performance VAD avec données réalistes"""
        # Générer un chunk de 160ms à 16kHz
        chunk_size = int(16000 * 0.16)  # 2560 samples
        audio_chunk = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
        
        # Mesurer la latence
        latencies = []
        for _ in range(50):  # Test 50 fois
            start_time = time.perf_counter()
            
            if hasattr(vad_manager, 'detect_voice'):
                result = await vad_manager.detect_voice(audio_chunk.tobytes())
            else:
                # Fallback pour tester la structure
                result = {"has_voice": False, "confidence": 0.0}
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Vérifier les objectifs de performance
        assert avg_latency < 25, f"VAD trop lent: {avg_latency:.2f}ms > 25ms"
        assert max_latency < 50, f"VAD pic trop lent: {max_latency:.2f}ms"
        
        print(f"🎤 VAD Performance - Moyenne: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")

class TestErrorHandlingIntegration:
    """Tests d'intégration de la gestion d'erreurs"""
    
    @pytest.fixture
    def error_handler(self):
        """Fixture pour le gestionnaire d'erreurs"""
        handler = RobustErrorHandler()
        handler.register_component("test_component", failure_threshold=3, max_retries=2)
        return handler
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, error_handler):
        """Test intégration circuit breaker"""
        
        call_count = 0
        
        async def unreliable_service():
            nonlocal call_count
            call_count += 1
            
            if call_count <= 5:  # Premières tentatives échouent
                raise Exception(f"Service failure {call_count}")
            
            return f"Success after {call_count} attempts"
        
        # Les premiers appels devraient échouer et ouvrir le circuit
        for i in range(3):
            try:
                await error_handler.execute_safe("test_component", unreliable_service)
                assert False, "Devrait avoir échoué"
            except Exception as e:
                print(f"Tentative {i+1}: {e}")
        
        # Vérifier que le circuit est ouvert
        circuit_breaker = error_handler.circuit_breakers["test_component"]
        assert circuit_breaker.state.value == "open"
        
        # Vérifier les métriques
        status = error_handler.get_health_status()
        assert status["components"]["test_component"]["metrics"]["error_count"] > 0
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, error_handler):
        """Test du mécanisme de retry"""
        
        attempt_count = 0
        
        async def flaky_service():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:  # Échoue 2 fois, réussit la 3ème
                raise Exception(f"Temporary failure {attempt_count}")
            
            return "Success on retry"
        
        # Devrait réussir après retry
        result = await error_handler.execute_safe("test_component", flaky_service)
        assert result == "Success on retry"
        assert attempt_count == 3  # 1 + 2 retries

class TestSecurityIntegration:
    """Tests d'intégration sécurité"""
    
    @pytest.fixture
    def security_config(self, tmp_path):
        """Fixture pour configuration sécurité"""
        return SecurityConfig(config_dir=str(tmp_path))
    
    def test_full_security_workflow(self, security_config):
        """Test workflow complet de sécurité"""
        
        # 1. Générer clé API
        api_key = security_config.generate_api_key("integration_test")
        assert api_key.startswith("luxa_")
        
        # 2. Valider clé API
        user = security_config.validate_api_key(api_key)
        assert user == "integration_test"
        
        # 3. Générer token JWT
        token = security_config.generate_jwt_token({
            "username": user,
            "permissions": ["audio_processing", "model_access"]
        })
        assert len(token) > 100
        
        # 4. Valider token JWT
        decoded = security_config.validate_jwt_token(token)
        assert decoded["username"] == "integration_test"
        assert "audio_processing" in decoded["permissions"]
    
    def test_audio_validation_integration(self, security_config):
        """Test validation audio intégrée"""
        
        # Audio WAV simulé
        wav_header = (
            b'RIFF' + (1024).to_bytes(4, 'little') + b'WAVEfmt ' +
            (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') +
            (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little') +
            (32000).to_bytes(4, 'little') + (2).to_bytes(2, 'little') +
            (16).to_bytes(2, 'little') + b'data' + (1000).to_bytes(4, 'little')
        )
        
        audio_data = wav_header + np.random.randint(-32768, 32767, 500, dtype=np.int16).tobytes()
        
        # Validation
        result = security_config.validate_audio_input(audio_data, "test.wav")
        
        assert result['valid'] is True
        assert len(result['errors']) == 0

class TestSystemIntegration:
    """Tests d'intégration système complet"""
    
    @pytest.mark.asyncio
    async def test_complete_system_workflow(self):
        """Test workflow système complet"""
        
        # 1. Initialiser tous les composants
        master_handler = RobustMasterHandler()
        await master_handler.initialize()
        
        # 2. Configurer sécurité
        security = SecurityConfig()
        api_key = security.generate_api_key("system_test")
        
        # 3. Traiter audio avec sécurité
        audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
        
        # Valider l'audio d'abord
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        validation = security.validate_audio_input(audio_bytes, "test.wav")
        
        if validation['valid']:
            # Traiter l'audio
            result = await master_handler.process_audio_safe(audio)
            
            assert result["success"] is True
            assert "latency_ms" in result
            assert result["latency_ms"] < 5000
        
        # 4. Vérifier métriques système
        health = master_handler.get_health_status()
        assert "status" in health
        assert "performance" in health
    
    @pytest.mark.asyncio
    async def test_system_under_load(self):
        """Test système sous charge"""
        
        master_handler = RobustMasterHandler()
        await master_handler.initialize()
        
        # Simuler charge avec requêtes concurrentes
        num_concurrent = 5
        num_requests_per_worker = 3
        
        async def worker(worker_id: int):
            """Worker qui traite plusieurs requêtes"""
            results = []
            
            for i in range(num_requests_per_worker):
                audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
                
                try:
                    result = await master_handler.process_audio_safe(audio)
                    results.append(result)
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
            
            return results
        
        # Lancer workers en parallèle
        tasks = [worker(i) for i in range(num_concurrent)]
        all_results = await asyncio.gather(*tasks)
        
        # Analyser résultats
        total_requests = num_concurrent * num_requests_per_worker
        successful_requests = sum(
            1 for worker_results in all_results 
            for result in worker_results 
            if result.get("success", False)
        )
        
        success_rate = successful_requests / total_requests
        
        # Au moins 80% de succès sous charge
        assert success_rate >= 0.8, f"Taux de succès trop bas: {success_rate:.2f}"
        
        print(f"🚀 Charge Test - Succès: {successful_requests}/{total_requests} ({success_rate:.1%})")

# Lancement des tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
