import pytest
import torch
import numpy as np
import sounddevice as sd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import time
from pathlib import Path
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from STT.stt_handler import STTHandler

class TestSTTHandler:
    """Tests unitaires pour STT/stt_handler.py avec coverage 80%"""
    
    @pytest.fixture
    def mock_config(self):
        """Configuration mock pour les tests"""
        return {
            'gpu_device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # RTX 3090 (CUDA:0) - NE PAS UTILISER CUDA:1 (RTX 5060)
            'sample_rate': 16000,
            'model_name': 'openai/whisper-base'
        }
    
    @pytest.fixture
    def stt_handler(self, mock_config):
        """Fixture STTHandler avec mocks"""
        with patch('STT.stt_handler.WhisperProcessor') as mock_processor, \
             patch('STT.stt_handler.WhisperForConditionalGeneration') as mock_model:
            
            # Configuration des mocks
            mock_processor_instance = Mock()
            mock_model_instance = Mock()
            
            mock_processor.from_pretrained.return_value = mock_processor_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            # Mock de la méthode to() du modèle
            mock_model_instance.to.return_value = mock_model_instance
            
            handler = STTHandler(mock_config)
            handler.processor = mock_processor_instance
            handler.model = mock_model_instance
            
            return handler
    
    @pytest.fixture
    def sample_audio_data(self):
        """Données audio simulées pour les tests"""
        # Génère un signal audio sinusoïdal simple
        duration = 2.0  # 2 secondes
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Signal sinusoïdal à 440Hz (note A4)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32).reshape(-1, 1)
    
    def test_init_with_cuda_available(self, mock_config):
        """Test initialisation avec CUDA disponible"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('STT.stt_handler.WhisperProcessor') as mock_processor, \
             patch('STT.stt_handler.WhisperForConditionalGeneration') as mock_model:
            
            mock_processor_instance = Mock()
            mock_model_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            handler = STTHandler(mock_config)
            
            assert handler.device == 'cuda:0'  # RTX 3090 (CUDA:0) UNIQUEMENT
            assert handler.sample_rate == 16000
            mock_model_instance.to.assert_called_with('cuda:0')  # RTX 3090 (CUDA:0) UNIQUEMENT
    
    def test_init_with_cuda_unavailable(self, mock_config):
        """Test initialisation avec CUDA non disponible"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('STT.stt_handler.WhisperProcessor') as mock_processor, \
             patch('STT.stt_handler.WhisperForConditionalGeneration') as mock_model:
            
            mock_processor_instance = Mock()
            mock_model_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            handler = STTHandler(mock_config)
            
            assert handler.device == 'cpu'
            mock_model_instance.to.assert_called_with('cpu')
    
    def test_init_model_loading_error(self, mock_config):
        """Test gestion d'erreur lors du chargement du modèle"""
        with patch('STT.stt_handler.WhisperProcessor') as mock_processor:
            mock_processor.from_pretrained.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception) as exc_info:
                STTHandler(mock_config)
            
            assert "Model loading failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_listen_and_transcribe_success(self, stt_handler, sample_audio_data):
        """Test transcription audio normale"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait') as mock_wait, \
             patch('torch.no_grad'):
            
            # Configuration des mocks
            mock_rec.return_value = sample_audio_data
            
            # Mock du processor
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            stt_handler.processor.return_value.input_features = mock_input_features
            
            # Mock du modèle
            mock_predicted_ids = Mock()
            stt_handler.model.generate.return_value = mock_predicted_ids
            stt_handler.processor.batch_decode.return_value = ["Bonjour le monde"]
            
            # Test
            start_time = time.time()
            result = stt_handler.listen_and_transcribe(duration=2)
            processing_time = time.time() - start_time
            
            # Vérifications
            assert result == "Bonjour le monde"
            assert processing_time < 2.0  # SLA performance <2s
            
            # Vérifier les appels
            mock_rec.assert_called_once_with(
                32000,  # 2 secondes * 16000 Hz
                samplerate=16000,
                channels=1,
                dtype='float32'
            )
            mock_wait.assert_called_once()
            stt_handler.model.generate.assert_called_once()
    
    def test_listen_and_transcribe_gpu_out_of_memory(self, stt_handler, sample_audio_data):
        """Test gestion d'erreur GPU out of memory"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            mock_rec.return_value = sample_audio_data
            
            # Mock du processor
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            stt_handler.processor.return_value.input_features = mock_input_features
            
            # Simuler erreur GPU
            stt_handler.model.generate.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
            
            with pytest.raises(torch.cuda.OutOfMemoryError) as exc_info:
                stt_handler.listen_and_transcribe(duration=2)
            
            assert "CUDA out of memory" in str(exc_info.value)
    
    def test_listen_and_transcribe_audio_processing_error(self, stt_handler):
        """Test gestion d'erreur pendant l'enregistrement audio"""
        with patch('sounddevice.rec') as mock_rec:
            mock_rec.side_effect = Exception("Audio device error")
            
            with pytest.raises(Exception) as exc_info:
                stt_handler.listen_and_transcribe(duration=2)
            
            assert "Audio device error" in str(exc_info.value)
    
    def test_listen_and_transcribe_empty_audio(self, stt_handler):
        """Test avec audio vide"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            # Audio vide
            empty_audio = np.array([]).reshape(-1, 1).astype(np.float32)
            mock_rec.return_value = empty_audio
            
            # Mock du processor
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            stt_handler.processor.return_value.input_features = mock_input_features
            
            # Mock du modèle
            mock_predicted_ids = Mock()
            stt_handler.model.generate.return_value = mock_predicted_ids
            stt_handler.processor.batch_decode.return_value = [""]
            
            result = stt_handler.listen_and_transcribe(duration=1)
            assert result == ""
    
    def test_listen_and_transcribe_very_quiet_audio(self, stt_handler):
        """Test avec audio très faible"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            # Audio très faible
            quiet_audio = np.random.normal(0, 0.001, (16000, 1)).astype(np.float32)
            mock_rec.return_value = quiet_audio
            
            # Mock du processor
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            stt_handler.processor.return_value.input_features = mock_input_features
            
            # Mock du modèle
            mock_predicted_ids = Mock()
            stt_handler.model.generate.return_value = mock_predicted_ids
            stt_handler.processor.batch_decode.return_value = [""]
            
            result = stt_handler.listen_and_transcribe(duration=1)
            assert isinstance(result, str)
    
    def test_listen_and_transcribe_different_durations(self, stt_handler):
        """Test avec différentes durées d'enregistrement"""
        durations = [1, 3, 5, 10]
        
        for duration in durations:
            with patch('sounddevice.rec') as mock_rec, \
                 patch('sounddevice.wait'), \
                 patch('torch.no_grad'):
                
                # Audio simulé pour la durée
                audio_length = int(duration * 16000)
                audio_data = np.random.normal(0, 0.1, (audio_length, 1)).astype(np.float32)
                mock_rec.return_value = audio_data
                
                # Mock du processor
                mock_input_features = Mock()
                mock_input_features.to.return_value = mock_input_features
                stt_handler.processor.return_value.input_features = mock_input_features
                
                # Mock du modèle
                mock_predicted_ids = Mock()
                stt_handler.model.generate.return_value = mock_predicted_ids
                stt_handler.processor.batch_decode.return_value = [f"Text for {duration}s"]
                
                result = stt_handler.listen_and_transcribe(duration=duration)
                assert result == f"Text for {duration}s"
                
                # Vérifier que la bonne durée est utilisée
                expected_samples = int(duration * 16000)
                mock_rec.assert_called_with(
                    expected_samples,
                    samplerate=16000,
                    channels=1,
                    dtype='float32'
                )
    
    def test_listen_and_transcribe_processor_timeout(self, stt_handler, sample_audio_data):
        """Test timeout du processor"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            mock_rec.return_value = sample_audio_data
            
            # Simuler timeout du processor
            stt_handler.processor.side_effect = TimeoutError("Processor timeout")
            
            with pytest.raises(TimeoutError) as exc_info:
                stt_handler.listen_and_transcribe(duration=2)
            
            assert "Processor timeout" in str(exc_info.value)
    
    def test_listen_and_transcribe_model_generate_error(self, stt_handler, sample_audio_data):
        """Test erreur lors de la génération du modèle"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            mock_rec.return_value = sample_audio_data
            
            # Mock du processor
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            stt_handler.processor.return_value.input_features = mock_input_features
            
            # Simuler erreur de génération
            stt_handler.model.generate.side_effect = RuntimeError("Model generation failed")
            
            with pytest.raises(RuntimeError) as exc_info:
                stt_handler.listen_and_transcribe(duration=2)
            
            assert "Model generation failed" in str(exc_info.value)
    
    def test_device_configuration(self, mock_config):
        """Test configuration du device selon disponibilité GPU"""
        with patch('STT.stt_handler.WhisperProcessor') as mock_processor, \
             patch('STT.stt_handler.WhisperForConditionalGeneration') as mock_model:
            
            mock_processor_instance = Mock()
            mock_model_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            # Test avec CUDA disponible
            with patch('torch.cuda.is_available', return_value=True):
                handler = STTHandler(mock_config)
                assert handler.device == mock_config['gpu_device']
            
            # Test avec CUDA non disponible
            with patch('torch.cuda.is_available', return_value=False):
                handler = STTHandler(mock_config)
                assert handler.device == 'cpu'
    
    def test_audio_preprocessing(self, stt_handler, sample_audio_data):
        """Test préprocessing de l'audio"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            mock_rec.return_value = sample_audio_data
            
            # Mock du processor avec vérification des paramètres
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            
            def mock_processor_call(*args, **kwargs):
                # Vérifier les paramètres passés au processor
                assert 'sampling_rate' in kwargs
                assert kwargs['sampling_rate'] == 16000
                assert 'return_tensors' in kwargs
                assert kwargs['return_tensors'] == 'pt'
                result = Mock()
                result.input_features = mock_input_features
                return result
            
            stt_handler.processor.side_effect = mock_processor_call
            
            # Mock du modèle
            mock_predicted_ids = Mock()
            stt_handler.model.generate.return_value = mock_predicted_ids
            stt_handler.processor.batch_decode.return_value = ["Test preprocessing"]
            
            result = stt_handler.listen_and_transcribe(duration=2)
            assert result == "Test preprocessing"
    
    @pytest.mark.performance
    def test_performance_benchmark(self, stt_handler, sample_audio_data):
        """Test benchmark de performance - SLA <2s"""
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            mock_rec.return_value = sample_audio_data
            
            # Mock optimisé pour la performance
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            stt_handler.processor.return_value.input_features = mock_input_features
            
            mock_predicted_ids = Mock()
            stt_handler.model.generate.return_value = mock_predicted_ids
            stt_handler.processor.batch_decode.return_value = ["Performance test"]
            
            # Mesure de performance
            start_time = time.time()
            result = stt_handler.listen_and_transcribe(duration=2)
            processing_time = time.time() - start_time
            
            # Vérifications SLA
            assert processing_time < 2.0, f"Processing time {processing_time:.2f}s exceeds SLA of 2s"
            assert result == "Performance test"
    
    def test_memory_usage_monitoring(self, stt_handler, sample_audio_data):
        """Test surveillance utilisation mémoire"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('sounddevice.rec') as mock_rec, \
             patch('sounddevice.wait'), \
             patch('torch.no_grad'):
            
            mock_rec.return_value = sample_audio_data
            
            # Mock du processor
            mock_input_features = Mock()
            mock_input_features.to.return_value = mock_input_features
            stt_handler.processor.return_value.input_features = mock_input_features
            
            # Mock du modèle
            mock_predicted_ids = Mock()
            stt_handler.model.generate.return_value = mock_predicted_ids
            stt_handler.processor.batch_decode.return_value = ["Memory test"]
            
            # Exécution
            result = stt_handler.listen_and_transcribe(duration=2)
            
            # Vérification mémoire
            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
            
            assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB too high"
            assert result == "Memory test"


class TestSTTHandlerIntegration:
    """Tests d'intégration STT avec composants externes"""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration pour tests d'intégration"""
        return {
            'gpu_device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # RTX 3090 (CUDA:0) - NE PAS UTILISER CUDA:1 (RTX 5060)
            'sample_rate': 16000,
            'model_name': 'openai/whisper-base',
            'vad_enabled': True
        }
    
    def test_stt_with_vad_integration(self, integration_config):
        """Test intégration STT avec VAD (simulation)"""
        # Ce test simule l'intégration avec le VAD Manager
        # En réalité, il faudrait une intégration plus poussée
        
        with patch('STT.stt_handler.WhisperProcessor') as mock_processor, \
             patch('STT.stt_handler.WhisperForConditionalGeneration') as mock_model:
            
            mock_processor_instance = Mock()
            mock_model_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            handler = STTHandler(integration_config)
            
            # Test que l'intégration VAD pourrait être ajoutée
            assert hasattr(handler, 'config')
            assert handler.config.get('vad_enabled') == True
    
    def test_concurrent_transcription_requests(self, integration_config):
        """Test requêtes de transcription concurrentes"""
        import concurrent.futures
        import threading
        
        with patch('STT.stt_handler.WhisperProcessor') as mock_processor, \
             patch('STT.stt_handler.WhisperForConditionalGeneration') as mock_model:
            
            mock_processor_instance = Mock()
            mock_model_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            handler = STTHandler(integration_config)
            
            def mock_transcribe(duration):
                with patch('sounddevice.rec') as mock_rec, \
                     patch('sounddevice.wait'), \
                     patch('torch.no_grad'):
                    
                    # Audio simulé
                    audio_length = int(duration * 16000)
                    audio_data = np.random.normal(0, 0.1, (audio_length, 1)).astype(np.float32)
                    mock_rec.return_value = audio_data
                    
                    # Mock du processor
                    mock_input_features = Mock()
                    mock_input_features.to.return_value = mock_input_features
                    handler.processor.return_value.input_features = mock_input_features
                    
                    # Mock du modèle
                    mock_predicted_ids = Mock()
                    handler.model.generate.return_value = mock_predicted_ids
                    handler.processor.batch_decode.return_value = [f"Concurrent {threading.current_thread().ident}"]
                    
                    return handler.listen_and_transcribe(duration=duration)
            
            # Test avec 3 requêtes concurrentes
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(mock_transcribe, 1) for _ in range(3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            assert len(results) == 3
            assert all("Concurrent" in result for result in results)


if __name__ == "__main__":
    # Exécution des tests avec coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=STT.stt_handler",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80"
    ])