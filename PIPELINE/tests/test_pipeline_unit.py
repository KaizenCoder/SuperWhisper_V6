#!/usr/bin/env python3
"""
Tests Unitaires Pipeline Orchestrator - Task 18.6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Tests unitaires pour validation composants pipeline :
- PipelineOrchestrator
- AudioOutputManager  
- LLMClient
- Fonctions utilitaires

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
import numpy as np
import wave
import io
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 Tests Pipeline: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Ajouter le répertoire parent au path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports du pipeline
from PIPELINE.pipeline_orchestrator import (
    PipelineOrchestrator,
    AudioOutputManager,
    LLMClient,
    _wav_bytes_to_numpy,
    _validate_rtx3090,
    PipelineMetrics,
    ConversationTurn
)

# =============================================================================
# FIXTURES ET MOCKS
# =============================================================================

@pytest.fixture
def mock_stt():
    """Mock STT manager"""
    stt = Mock()
    stt.initialized = True
    stt.initialize = AsyncMock()
    return stt

@pytest.fixture
def mock_tts():
    """Mock TTS manager"""
    tts = Mock()
    
    # Mock TTS result
    tts_result = Mock()
    tts_result.success = True
    tts_result.audio_data = create_test_wav_bytes()
    tts_result.error = None
    
    tts.synthesize = Mock(return_value=tts_result)
    return tts

@pytest.fixture
def sample_wav_bytes():
    """Créer des données WAV de test"""
    return create_test_wav_bytes()

def create_test_wav_bytes(duration_ms=500, sample_rate=22050):
    """Créer des données WAV de test"""
    # Générer un signal sinusoïdal simple
    duration_s = duration_ms / 1000.0
    samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, samples)
    frequency = 440  # La note A4
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convertir en int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Créer fichier WAV en mémoire
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    return buffer.getvalue()

# =============================================================================
# TESTS FONCTIONS UTILITAIRES
# =============================================================================

class TestUtilityFunctions:
    """Tests des fonctions utilitaires"""
    
    def test_wav_bytes_to_numpy_valid(self, sample_wav_bytes):
        """Test conversion WAV bytes vers numpy array"""
        audio = _wav_bytes_to_numpy(sample_wav_bytes)
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        assert np.all(audio >= -1.0) and np.all(audio <= 1.0)
    
    def test_wav_bytes_to_numpy_invalid(self):
        """Test conversion avec données invalides"""
        # Données corrompues
        invalid_data = b"invalid wav data"
        audio = _wav_bytes_to_numpy(invalid_data)
        
        # Doit retourner du silence
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.allclose(audio, 0.0)
    
    def test_wav_bytes_to_numpy_empty(self):
        """Test conversion avec données vides"""
        audio = _wav_bytes_to_numpy(b"")
        
        # Doit retourner du silence
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.allclose(audio, 0.0)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_validate_rtx3090_success(self, mock_props, mock_cuda_available):
        """Test validation RTX 3090 réussie"""
        mock_cuda_available.return_value = True
        
        # Mock propriétés GPU
        gpu_props = Mock()
        gpu_props.name = "NVIDIA GeForce RTX 3090"
        gpu_props.total_memory = 24 * 1024**3  # 24GB
        mock_props.return_value = gpu_props
        
        # Doit passer sans exception
        _validate_rtx3090()
    
    @patch('torch.cuda.is_available')
    def test_validate_rtx3090_no_cuda(self, mock_cuda_available):
        """Test validation RTX 3090 sans CUDA"""
        mock_cuda_available.return_value = False
        
        with pytest.raises(RuntimeError, match="CUDA not available"):
            _validate_rtx3090()
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_validate_rtx3090_insufficient_vram(self, mock_props, mock_cuda_available):
        """Test validation RTX 3090 VRAM insuffisante"""
        mock_cuda_available.return_value = True
        
        # Mock GPU avec VRAM insuffisante
        gpu_props = Mock()
        gpu_props.name = "NVIDIA GeForce RTX 5060"
        gpu_props.total_memory = 8 * 1024**3  # 8GB seulement
        mock_props.return_value = gpu_props
        
        with pytest.raises(RuntimeError, match="GPU VRAM insufficient"):
            _validate_rtx3090()

# =============================================================================
# TESTS AUDIO OUTPUT MANAGER
# =============================================================================

class TestAudioOutputManager:
    """Tests AudioOutputManager"""
    
    def test_audio_output_manager_init(self):
        """Test initialisation AudioOutputManager"""
        manager = AudioOutputManager(sample_rate=22050, channels=1)
        
        assert manager.sample_rate == 22050
        assert manager.channels == 1
        assert not manager._running.is_set()
    
    @patch('simpleaudio.play_buffer')
    def test_play_async_valid_audio(self, mock_play_buffer):
        """Test lecture audio asynchrone"""
        manager = AudioOutputManager()
        
        # Mock play object
        play_obj = Mock()
        play_obj.wait_done = Mock()
        mock_play_buffer.return_value = play_obj
        
        # Audio de test
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        manager.play_async(audio)
        
        # Attendre que le worker traite
        import time
        time.sleep(0.2)
        
        # Vérifier que play_buffer a été appelé
        assert mock_play_buffer.called
    
    def test_play_async_empty_audio(self):
        """Test lecture audio vide"""
        manager = AudioOutputManager()
        
        # Audio vide
        audio = np.array([], dtype=np.float32)
        
        # Ne doit pas lever d'exception
        manager.play_async(audio)
    
    def test_stop(self):
        """Test arrêt AudioOutputManager"""
        manager = AudioOutputManager()
        
        # Démarrer puis arrêter
        audio = np.array([0.1, 0.2], dtype=np.float32)
        manager.play_async(audio)
        manager.stop()
        
        assert not manager._running.is_set()

# =============================================================================
# TESTS LLM CLIENT
# =============================================================================

class TestLLMClient:
    """Tests LLMClient"""
    
    def test_llm_client_init(self):
        """Test initialisation LLMClient"""
        client = LLMClient("http://localhost:8000", model="test-model")
        
        assert client.endpoint == "http://localhost:8000/v1/chat/completions"
        assert client.model == "test-model"
    
    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test génération LLM réussie"""
        client = LLMClient("http://localhost:8000")
        
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        
        with patch.object(client._client, 'post', return_value=mock_response) as mock_post:
            mock_post.return_value = mock_response
            
            result = await client.generate("Test prompt", [])
            
            assert result == "Test response"
            assert mock_post.called
    
    @pytest.mark.asyncio
    async def test_generate_fallback(self):
        """Test fallback LLM"""
        client = LLMClient("http://localhost:8000")
        
        # Mock exception
        with patch.object(client._client, 'post', side_effect=Exception("Connection error")):
            result = await client.generate("bonjour", [])
            
            # Doit utiliser le fallback
            assert "bonjour" in result.lower() or "comment puis-je" in result.lower()
    
    def test_fallback_responses(self):
        """Test réponses fallback"""
        # Test différents types de prompts
        assert "Il est" in LLMClient._fallback("quelle heure est-il")
        assert "Bonjour" in LLMClient._fallback("bonjour")
        assert "De rien" in LLMClient._fallback("merci")
        assert "J'ai entendu" in LLMClient._fallback("autre chose")

# =============================================================================
# TESTS PIPELINE ORCHESTRATOR
# =============================================================================

class TestPipelineOrchestrator:
    """Tests PipelineOrchestrator"""
    
    @patch('PIPELINE.pipeline_orchestrator._validate_rtx3090')
    def test_pipeline_orchestrator_init(self, mock_validate, mock_stt, mock_tts):
        """Test initialisation PipelineOrchestrator"""
        orchestrator = PipelineOrchestrator(
            stt=mock_stt,
            tts=mock_tts,
            llm_endpoint="http://localhost:8000",
            metrics_enabled=False
        )
        
        assert orchestrator.stt == mock_stt
        assert orchestrator.tts == mock_tts
        assert isinstance(orchestrator.llm, LLMClient)
        assert isinstance(orchestrator.audio_out, AudioOutputManager)
        assert mock_validate.called
    
    @patch('PIPELINE.pipeline_orchestrator._validate_rtx3090')
    def test_on_transcription(self, mock_validate, mock_stt, mock_tts):
        """Test callback transcription"""
        orchestrator = PipelineOrchestrator(mock_stt, mock_tts)
        
        # Test transcription
        orchestrator._on_transcription("test text", 100.0)
        
        # Vérifier que la queue contient le texte
        assert not orchestrator._text_q.empty()
        text, latency = orchestrator._text_q.get_nowait()
        assert text == "test text"
        assert latency == 100.0
    
    @patch('PIPELINE.pipeline_orchestrator._validate_rtx3090')
    def test_get_metrics(self, mock_validate, mock_stt, mock_tts):
        """Test récupération métriques"""
        orchestrator = PipelineOrchestrator(mock_stt, mock_tts)
        
        metrics = orchestrator.get_metrics()
        
        assert isinstance(metrics, PipelineMetrics)
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
    
    @patch('PIPELINE.pipeline_orchestrator._validate_rtx3090')
    def test_get_conversation_history(self, mock_validate, mock_stt, mock_tts):
        """Test récupération historique conversation"""
        orchestrator = PipelineOrchestrator(mock_stt, mock_tts)
        
        # Ajouter un tour de conversation
        turn = ConversationTurn(
            user_text="test",
            assistant_text="response",
            total_latency_ms=1000.0,
            stt_latency_ms=300.0,
            llm_latency_ms=400.0,
            tts_latency_ms=300.0,
            audio_latency_ms=0.0
        )
        orchestrator._history.append(turn)
        
        history = orchestrator.get_conversation_history(max_turns=5)
        
        assert len(history) == 1
        assert history[0].user_text == "test"
        assert history[0].assistant_text == "response"

# =============================================================================
# TESTS DATA CLASSES
# =============================================================================

class TestDataClasses:
    """Tests des classes de données"""
    
    def test_pipeline_metrics(self):
        """Test PipelineMetrics"""
        metrics = PipelineMetrics()
        
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_latency_ms == 0.0
    
    def test_conversation_turn(self):
        """Test ConversationTurn"""
        turn = ConversationTurn(
            user_text="Hello",
            assistant_text="Hi there",
            total_latency_ms=1000.0,
            stt_latency_ms=300.0,
            llm_latency_ms=400.0,
            tts_latency_ms=300.0,
            audio_latency_ms=0.0
        )
        
        assert turn.user_text == "Hello"
        assert turn.assistant_text == "Hi there"
        assert turn.success == True
        assert turn.error is None

# =============================================================================
# TESTS INTÉGRATION WORKERS
# =============================================================================

class TestWorkers:
    """Tests des workers asynchrones"""
    
    @pytest.mark.asyncio
    @patch('PIPELINE.pipeline_orchestrator._validate_rtx3090')
    async def test_llm_worker_processing(self, mock_validate, mock_stt, mock_tts):
        """Test traitement LLM worker"""
        orchestrator = PipelineOrchestrator(mock_stt, mock_tts)
        
        # Mock LLM generate
        orchestrator.llm.generate = AsyncMock(return_value="Test response")
        
        # Ajouter texte à la queue
        await orchestrator._text_q.put(("test input", 100.0))
        
        # Créer et démarrer worker
        worker_task = asyncio.create_task(orchestrator._llm_worker())
        
        # Attendre traitement
        await asyncio.sleep(0.1)
        
        # Vérifier que la response queue contient le résultat
        assert not orchestrator._response_q.empty()
        
        # Nettoyer
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    @patch('PIPELINE.pipeline_orchestrator._validate_rtx3090')
    async def test_tts_worker_processing(self, mock_validate, mock_stt, mock_tts):
        """Test traitement TTS worker"""
        orchestrator = PipelineOrchestrator(mock_stt, mock_tts)
        
        # Ajouter réponse à la queue
        await orchestrator._response_q.put(("user text", "assistant text", 100.0, 200.0))
        
        # Créer et démarrer worker
        worker_task = asyncio.create_task(orchestrator._tts_worker())
        
        # Attendre traitement
        await asyncio.sleep(0.1)
        
        # Vérifier que l'historique contient le tour
        assert len(orchestrator._history) > 0
        
        # Nettoyer
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

# =============================================================================
# CONFIGURATION PYTEST
# =============================================================================

if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v", "--tb=short"]) 