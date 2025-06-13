#!/usr/bin/env python3
"""
Tests d'intégration PrismSTTBackend avec faster-whisper - RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🧪 Tests Prism Integration - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import pytest
import numpy as np
import asyncio
import time
import torch
from pathlib import Path

# Import du backend STT
sys.path.append(str(Path(__file__).parent.parent))
from STT.backends.prism_stt_backend import PrismSTTBackend, create_prism_backend, validate_rtx3090_stt

def generate_test_audio(duration=5.0, sample_rate=16000, audio_type="speech_simulation"):
    """
    Génère différents types d'audio test pour validation.
    
    Args:
        duration: Durée en secondes
        sample_rate: Fréquence d'échantillonnage
        audio_type: Type d'audio ("silence", "noise", "speech_simulation", "tone")
    
    Returns:
        np.ndarray: Audio test
    """
    samples = int(sample_rate * duration)
    
    if audio_type == "silence":
        # Silence complet
        audio = np.zeros(samples, dtype=np.float32)
        
    elif audio_type == "noise":
        # Bruit blanc
        audio = np.random.normal(0, 0.1, samples).astype(np.float32)
        
    elif audio_type == "tone":
        # Ton pur 440Hz (La)
        t = np.linspace(0, duration, samples)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
    elif audio_type == "speech_simulation":
        # Simulation parole avec variations fréquentielles
        t = np.linspace(0, duration, samples)
        
        # Fréquences fondamentales typiques de la parole
        f1 = 150 + 50 * np.sin(2 * np.pi * 2 * t)  # Variation 100-200Hz
        f2 = 800 + 200 * np.sin(2 * np.pi * 3 * t)  # Variation 600-1000Hz
        f3 = 2400 + 400 * np.sin(2 * np.pi * 1.5 * t)  # Variation 2000-2800Hz
        
        # Combinaison des harmoniques
        audio = (0.4 * np.sin(2 * np.pi * f1 * t) +
                0.3 * np.sin(2 * np.pi * f2 * t) +
                0.2 * np.sin(2 * np.pi * f3 * t))
        
        # Enveloppe pour simuler des mots
        envelope = np.ones_like(t)
        for i in range(int(duration)):
            start = i * sample_rate
            end = min((i + 0.8) * sample_rate, len(envelope))
            envelope[int(start):int(end)] *= np.hanning(int(end - start))
        
        audio *= envelope
        audio = audio.astype(np.float32)
        
        # Ajout de bruit léger
        noise = np.random.normal(0, 0.02, samples).astype(np.float32)
        audio += noise
        
    else:
        raise ValueError(f"Type audio non supporté: {audio_type}")
    
    # Normalisation
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio

class TestPrismIntegration:
    """Tests d'intégration PrismSTTBackend"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup pour chaque test"""
        # Validation GPU obligatoire
        validate_rtx3090_stt()
        
        # Configuration test
        self.config_large = {
            'model': 'large-v2',
            'compute_type': 'float16',
            'language': 'fr',
            'vad_filter': True,
            'beam_size': 5,
            'best_of': 5
        }
        
        self.config_tiny = {
            'model': 'tiny',
            'compute_type': 'float16',
            'language': 'fr',
            'vad_filter': False,  # Plus rapide pour tiny
            'beam_size': 1,
            'best_of': 1
        }
    
    def test_gpu_configuration_validation(self):
        """Test validation configuration GPU RTX 3090"""
        # Validation environnement
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
        assert torch.cuda.is_available()
        
        # Validation GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🎮 GPU détectée: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.1f}GB")
        
        # Assertions RTX 3090
        assert "3090" in gpu_name, f"RTX 3090 requise, détectée: {gpu_name}"
        assert gpu_memory > 20, f"VRAM insuffisante: {gpu_memory:.1f}GB < 20GB"
    
    def test_backend_initialization_large(self):
        """Test initialisation backend large-v2"""
        backend = PrismSTTBackend(self.config_large)
        
        # Validations
        assert backend.model_loaded
        assert backend.model_size == 'large-v2'
        assert backend.device == 'cuda:0'
        assert backend.compute_type == 'float16'
        
        # Métriques initiales
        metrics = backend.get_metrics()
        assert metrics['model_size'] == 'large-v2'
        assert metrics['device'] == 'cuda:0'
        assert metrics['total_requests'] == 0
        
        # Health check
        assert backend.health_check()
        
        # Nettoyage
        backend.cleanup()
        print("✅ Backend large-v2 initialisé avec succès")
    
    def test_backend_initialization_tiny(self):
        """Test initialisation backend tiny"""
        backend = PrismSTTBackend(self.config_tiny)
        
        # Validations
        assert backend.model_loaded
        assert backend.model_size == 'tiny'
        assert backend.device == 'cuda:0'
        
        # Health check
        assert backend.health_check()
        
        # Nettoyage
        backend.cleanup()
        print("✅ Backend tiny initialisé avec succès")
    
    @pytest.mark.asyncio
    async def test_transcription_silence(self):
        """Test transcription audio silence"""
        backend = create_prism_backend("tiny")  # Plus rapide pour test
        
        # Audio silence 3 secondes
        audio = generate_test_audio(duration=3.0, audio_type="silence")
        
        # Transcription
        start_time = time.time()
        result = await backend.transcribe(audio)
        latency = time.time() - start_time
        
        # Validations
        assert result.success, f"Transcription échouée: {result.error}"
        assert result.processing_time < 2.0, f"Trop lent: {result.processing_time:.2f}s"
        assert result.rtf < 1.0, f"RTF {result.rtf:.2f} > 1.0"
        assert result.device == "cuda:0"
        assert result.backend_used == "prism_tiny"
        
        print(f"✅ Transcription silence:")
        print(f"   Texte: '{result.text}'")
        print(f"   Latence: {latency*1000:.0f}ms")
        print(f"   RTF: {result.rtf:.2f}")
        print(f"   Confiance: {result.confidence:.2f}")
        
        backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_transcription_speech_simulation(self):
        """Test transcription simulation parole"""
        backend = create_prism_backend("large-v2")
        
        # Audio simulation parole 5 secondes
        audio = generate_test_audio(duration=5.0, audio_type="speech_simulation")
        
        # Transcription
        start_time = time.time()
        result = await backend.transcribe(audio)
        latency = time.time() - start_time
        
        # Validations performance RTX 3090
        assert result.success, f"Transcription échouée: {result.error}"
        assert result.processing_time < 3.0, f"Trop lent: {result.processing_time:.2f}s"
        assert result.rtf < 0.8, f"RTF {result.rtf:.2f} > 0.8"
        assert result.device == "cuda:0"
        assert result.backend_used == "prism_large-v2"
        
        # Validation segments
        assert isinstance(result.segments, list)
        
        print(f"✅ Transcription simulation parole:")
        print(f"   Texte: '{result.text}'")
        print(f"   Latence: {latency*1000:.0f}ms")
        print(f"   RTF: {result.rtf:.2f}")
        print(f"   Segments: {len(result.segments)}")
        
        backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_transcription_multiple_durations(self):
        """Test transcription différentes durées"""
        backend = create_prism_backend("tiny")  # Plus rapide
        
        durations = [1.0, 3.0, 5.0, 10.0]
        results = []
        
        for duration in durations:
            audio = generate_test_audio(duration=duration, audio_type="speech_simulation")
            
            start_time = time.time()
            result = await backend.transcribe(audio)
            latency = time.time() - start_time
            
            # Validations
            assert result.success, f"Échec pour {duration}s: {result.error}"
            assert result.rtf < 1.0, f"RTF {result.rtf:.2f} > 1.0 pour {duration}s"
            
            results.append({
                'duration': duration,
                'latency': latency,
                'rtf': result.rtf,
                'text_length': len(result.text)
            })
            
            print(f"✅ {duration}s: {latency*1000:.0f}ms (RTF: {result.rtf:.2f})")
        
        # Validation tendance RTF
        rtfs = [r['rtf'] for r in results]
        assert all(rtf < 1.0 for rtf in rtfs), "Tous les RTF doivent être < 1.0"
        
        backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_gpu_memory_monitoring(self):
        """Test surveillance mémoire GPU"""
        backend = create_prism_backend("large-v2")
        
        # Mémoire avant transcription
        memory_before = backend.get_gpu_memory_usage()
        
        # Transcription
        audio = generate_test_audio(duration=5.0, audio_type="speech_simulation")
        result = await backend.transcribe(audio)
        
        # Mémoire après transcription
        memory_after = backend.get_gpu_memory_usage()
        
        # Validations mémoire
        assert memory_before['total_gb'] > 20, "RTX 3090 doit avoir >20GB"
        assert memory_after['usage_percent'] < 90, "Utilisation VRAM < 90%"
        assert memory_after['free_gb'] > 2, "Au moins 2GB libre"
        
        print(f"💾 Mémoire GPU:")
        print(f"   Total: {memory_after['total_gb']:.1f}GB")
        print(f"   Utilisée: {memory_after['usage_percent']:.1f}%")
        print(f"   Libre: {memory_after['free_gb']:.1f}GB")
        
        backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_backend_metrics(self):
        """Test métriques backend"""
        backend = create_prism_backend("tiny")
        
        # Plusieurs transcriptions
        for i in range(3):
            audio = generate_test_audio(duration=2.0, audio_type="speech_simulation")
            result = await backend.transcribe(audio)
            assert result.success
        
        # Métriques
        metrics = backend.get_metrics()
        
        # Validations
        assert metrics['total_requests'] == 3
        assert metrics['total_errors'] == 0
        assert metrics['error_rate'] == 0.0
        assert metrics['average_latency'] > 0
        assert 'gpu_memory' in metrics
        
        print(f"📊 Métriques backend:")
        print(f"   Requêtes: {metrics['total_requests']}")
        print(f"   Erreurs: {metrics['total_errors']}")
        print(f"   Latence moyenne: {metrics['average_latency']:.3f}s")
        print(f"   Prism disponible: {metrics['prism_available']}")
        
        backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test gestion d'erreurs"""
        backend = create_prism_backend("tiny")
        
        # Audio invalide (vide)
        invalid_audio = np.array([], dtype=np.float32)
        
        result = await backend.transcribe(invalid_audio)
        
        # Validation gestion erreur
        assert not result.success
        assert result.error is not None
        assert result.rtf == 999.0
        assert result.text == ""
        
        print(f"✅ Gestion erreur validée: {result.error}")
        
        backend.cleanup()
    
    def test_create_prism_backend_utility(self):
        """Test fonction utilitaire create_prism_backend"""
        # Backend par défaut
        backend1 = create_prism_backend()
        assert backend1.model_size == "large-v2"
        assert backend1.compute_type == "float16"
        backend1.cleanup()
        
        # Backend personnalisé
        backend2 = create_prism_backend(
            model_size="tiny",
            compute_type="int8",
            language="en"
        )
        assert backend2.model_size == "tiny"
        assert backend2.compute_type == "int8"
        assert backend2.language == "en"
        backend2.cleanup()
        
        print("✅ Fonction utilitaire create_prism_backend validée")

# Tests de performance spécifiques RTX 3090
class TestPrismPerformanceRTX3090:
    """Tests de performance spécifiques RTX 3090"""
    
    @pytest.mark.asyncio
    async def test_performance_large_v2_rtx3090(self):
        """Test performance large-v2 sur RTX 3090"""
        backend = create_prism_backend("large-v2")
        
        # Audio test 5 secondes
        audio = generate_test_audio(duration=5.0, audio_type="speech_simulation")
        
        # Mesure performance
        start_time = time.time()
        result = await backend.transcribe(audio)
        total_time = time.time() - start_time
        
        # Objectifs performance RTX 3090
        assert result.success
        assert result.rtf < 0.5, f"RTF {result.rtf:.2f} > 0.5 (objectif RTX 3090)"
        assert total_time < 2.5, f"Latence {total_time:.2f}s > 2.5s"
        
        print(f"🏆 Performance large-v2 RTX 3090:")
        print(f"   RTF: {result.rtf:.2f} (objectif < 0.5)")
        print(f"   Latence: {total_time*1000:.0f}ms")
        print(f"   Texte: '{result.text[:50]}...'")
        
        backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_tiny_rtx3090(self):
        """Test performance tiny sur RTX 3090"""
        backend = create_prism_backend("tiny")
        
        # Audio test 3 secondes
        audio = generate_test_audio(duration=3.0, audio_type="speech_simulation")
        
        # Mesure performance
        start_time = time.time()
        result = await backend.transcribe(audio)
        total_time = time.time() - start_time
        
        # Objectifs performance tiny RTX 3090
        assert result.success
        assert result.rtf < 0.2, f"RTF {result.rtf:.2f} > 0.2 (objectif tiny RTX 3090)"
        assert total_time < 1.0, f"Latence {total_time:.2f}s > 1.0s"
        
        print(f"⚡ Performance tiny RTX 3090:")
        print(f"   RTF: {result.rtf:.2f} (objectif < 0.2)")
        print(f"   Latence: {total_time*1000:.0f}ms")
        
        backend.cleanup()

# Point d'entrée pour tests manuels
if __name__ == "__main__":
    print("🧪 Tests d'intégration PrismSTTBackend - RTX 3090")
    
    # Test rapide
    async def quick_test():
        print("\n🚀 Test rapide PrismSTTBackend...")
        
        backend = create_prism_backend("tiny")
        audio = generate_test_audio(duration=2.0, audio_type="speech_simulation")
        
        result = await backend.transcribe(audio)
        
        print(f"✅ Succès: {result.success}")
        print(f"📝 Texte: '{result.text}'")
        print(f"⏱️ Latence: {result.processing_time*1000:.0f}ms")
        print(f"📊 RTF: {result.rtf:.2f}")
        print(f"🎮 Device: {result.device}")
        
        backend.cleanup()
    
    # Exécution test rapide
    asyncio.run(quick_test())
    
    print("\n🎯 Pour tests complets: pytest tests/test_prism_integration.py -v") 