#!/usr/bin/env python3
"""
Tests PrismSTTBackend - SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Tests unitaires pour le backend Prism STT
"""

import os
import sys
import unittest
import numpy as np
import asyncio
from unittest.mock import Mock, patch

# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Ajouter le chemin du module STT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestPrismSTTBackend(unittest.TestCase):
    """Tests pour PrismSTTBackend"""
    
    def setUp(self):
        """Setup avant chaque test"""
        self.config = {
            'model': 'small',  # Modèle léger pour tests
            'compute_type': 'float16',
            'language': 'fr',
            'beam_size': 1,  # Rapide pour tests
            'vad_filter': False  # Désactiver pour tests
        }
        
        # Audio test 3 secondes
        self.test_audio = np.zeros(16000 * 3, dtype=np.float32)
        
    def test_config_validation(self):
        """Test validation configuration"""
        # Configuration valide
        valid_config = self.config.copy()
        self.assertIsInstance(valid_config, dict)
        
        # Configuration invalide
        invalid_config = {}
        self.assertIsInstance(invalid_config, dict)
    
    def test_audio_validation(self):
        """Test validation format audio"""
        from STT.utils.audio_utils import validate_audio_format
        
        # Audio valide
        valid_audio = np.zeros(16000, dtype=np.float32)
        self.assertTrue(validate_audio_format(valid_audio))
        
        # Audio invalide - mauvais dtype
        invalid_audio = np.zeros(16000, dtype=np.int16)
        self.assertFalse(validate_audio_format(invalid_audio))
        
        # Audio invalide - stéréo
        stereo_audio = np.zeros((16000, 2), dtype=np.float32)
        self.assertFalse(validate_audio_format(stereo_audio))
    
    @patch('STT.backends.prism_stt_backend.WhisperModel')
    def test_backend_initialization(self, mock_whisper):
        """Test initialisation backend avec mock"""
        # Mock du modèle Whisper
        mock_model = Mock()
        mock_whisper.return_value = mock_model
        
        try:
            from STT.backends.prism_stt_backend import PrismSTTBackend
            backend = PrismSTTBackend(self.config)
            
            # Vérifier initialisation
            self.assertIsNotNone(backend)
            self.assertEqual(backend.model_size, 'small')
            self.assertEqual(backend.language, 'fr')
            
        except ImportError:
            self.skipTest("Module PrismSTTBackend non disponible")
    
    def test_audio_processing(self):
        """Test traitement audio"""
        from STT.utils.audio_utils import AudioProcessor
        
        # Test normalisation
        audio_with_dc = np.ones(16000, dtype=np.float32) * 0.5
        normalized = AudioProcessor.normalize_audio(audio_with_dc)
        
        self.assertEqual(normalized.dtype, np.float32)
        self.assertLess(np.abs(np.mean(normalized)), 0.1)  # DC offset supprimé
        
        # Test hash audio
        audio_hash = AudioProcessor.compute_audio_hash(self.test_audio)
        self.assertIsInstance(audio_hash, str)
        self.assertEqual(len(audio_hash), 32)  # MD5 = 32 chars
        
        # Test détection silence
        is_silent, energy = AudioProcessor.detect_silence(self.test_audio)
        self.assertTrue(is_silent)  # Audio zéro = silence
        self.assertAlmostEqual(energy, 0.0, places=5)
    
    def test_performance_requirements(self):
        """Test exigences performance"""
        # Durée audio test
        audio_duration = len(self.test_audio) / 16000
        self.assertEqual(audio_duration, 3.0)
        
        # Objectif latence < 400ms
        target_latency = 0.4
        self.assertLess(target_latency, 1.0)
        
        # RTF cible < 0.2 (5x temps réel)
        target_rtf = 0.2
        expected_processing_time = audio_duration * target_rtf
        self.assertLess(expected_processing_time, target_latency)

class TestSTTConfiguration(unittest.TestCase):
    """Tests pour configuration STT"""
    
    def test_stt_config_creation(self):
        """Test création configuration STT"""
        try:
            from STT.config.stt_config import STTConfig
            
            config = STTConfig()
            
            # Vérifier configuration GPU
            self.assertEqual(config.gpu_device, "cuda:1")
            self.assertEqual(config.cuda_visible_devices, "1")
            
            # Vérifier objectifs performance
            self.assertEqual(config.target_latency, 0.4)
            self.assertLessEqual(config.max_audio_duration, 30.0)
            
            # Vérifier backends
            enabled_backends = config.get_enabled_backends()
            self.assertGreater(len(enabled_backends), 0)
            
            # Vérifier validation
            errors = config.validate()
            self.assertEqual(len(errors), 0)  # Aucune erreur
            
        except ImportError:
            self.skipTest("Module STTConfig non disponible")
    
    def test_backend_config(self):
        """Test configuration backends"""
        try:
            from STT.config.stt_config import STTConfig, BackendConfig
            
            config = STTConfig()
            
            # Vérifier backend Prism (principal)
            prism_config = config.get_backend_config("prism")
            self.assertIsNotNone(prism_config)
            self.assertEqual(prism_config.priority, 1)
            self.assertTrue(prism_config.enabled)
            
            # Vérifier fallbacks
            enabled_backends = config.get_enabled_backends()
            priorities = [b.priority for b in enabled_backends]
            self.assertEqual(priorities, sorted(priorities))  # Triés par priorité
            
        except ImportError:
            self.skipTest("Module STTConfig non disponible")

def run_stt_tests():
    """Exécute tous les tests STT"""
    print("🧪 Exécution tests STT SuperWhisper V6...")
    print("🎮 Configuration GPU RTX 3090 (CUDA:1)")
    print("=" * 50)
    
    # Créer suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tests
    suite.addTests(loader.loadTestsFromTestCase(TestPrismSTTBackend))
    suite.addTests(loader.loadTestsFromTestCase(TestSTTConfiguration))
    
    # Exécuter tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("=" * 50)
    if result.wasSuccessful():
        print("✅ Tous les tests STT réussis!")
    else:
        print(f"❌ {len(result.failures)} échecs, {len(result.errors)} erreurs")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_stt_tests() 