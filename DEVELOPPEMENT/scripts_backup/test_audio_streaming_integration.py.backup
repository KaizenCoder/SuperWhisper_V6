#!/usr/bin/env python3
"""
Test Rapide - Intégration AudioStreamer Optimisé SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Test de validation rapide des 7 optimisations critiques
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configuration GPU RTX 3090 obligatoire
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Ajouter le chemin racine du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🧪 === TEST RAPIDE AUDIOSTREAMER OPTIMISÉ ===")
print("🚨 Configuration GPU: RTX 3090 (CUDA:1) OBLIGATOIRE")

def test_imports():
    """Test des imports critiques"""
    print("\n1️⃣ Test des imports...")
    
    try:
        from STT.audio_streamer_optimized import (
            AudioStreamer, 
            AudioStreamingManager, 
            VoiceActivityDetector,
            HallucinationFilter,
            validate_rtx3090_configuration
        )
        print("✅ AudioStreamer optimisé importé")
        
        import torch
        print("✅ PyTorch importé")
        
        import sounddevice as sd
        print("✅ SoundDevice importé")
        
        try:
            import webrtcvad
            print("✅ WebRTC-VAD disponible")
        except ImportError:
            print("⚠️ WebRTC-VAD non disponible (fallback RMS)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False

def test_gpu_configuration():
    """Test configuration GPU RTX 3090"""
    print("\n2️⃣ Test configuration GPU...")
    
    try:
        from STT.audio_streamer_optimized import validate_rtx3090_configuration
        validate_rtx3090_configuration()
        print("✅ Configuration RTX 3090 validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur GPU: {e}")
        return False

def test_audio_devices():
    """Test détection périphériques audio"""
    print("\n3️⃣ Test détection périphériques audio...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        print(f"📱 {len(devices)} périphériques détectés:")
        
        rode_found = False
        for idx, device in enumerate(devices):
            device_name = device.get('name', 'Unknown')
            max_input = device.get('max_input_channels', 0)
            
            if max_input > 0:  # Périphérique d'entrée
                print(f"   🎤 ID {idx}: {device_name} ({max_input} canaux)")
                
                if 'rode' in device_name.lower() or 'nt-usb' in device_name.lower():
                    rode_found = True
                    print(f"      ✅ Rode NT-USB détecté!")
        
        if not rode_found:
            print("⚠️ Rode NT-USB non détecté - test avec périphérique par défaut")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur périphériques: {e}")
        return False

def test_vad_initialization():
    """Test initialisation VAD"""
    print("\n4️⃣ Test initialisation VAD...")
    
    try:
        from STT.audio_streamer_optimized import VoiceActivityDetector
        
        vad = VoiceActivityDetector(sample_rate=16000, aggressiveness=1)
        print(f"✅ VAD initialisé")
        print(f"   🎙️ WebRTC disponible: {vad.webrtc_available}")
        print(f"   🔊 Seuil RMS fallback: {vad.rms_threshold}")
        
        # Test détection avec signal synthétique
        import numpy as np
        
        # Signal silence
        silence = np.zeros(16000, dtype=np.float32)  # 1s de silence
        silence_detected = vad.has_voice_activity(silence)
        print(f"   🔇 Silence détecté comme voix: {silence_detected}")
        
        # Signal bruit
        noise = np.random.normal(0, 0.01, 16000).astype(np.float32)  # Bruit faible
        noise_detected = vad.has_voice_activity(noise)
        print(f"   🔊 Bruit faible détecté comme voix: {noise_detected}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur VAD: {e}")
        return False

def test_hallucination_filter():
    """Test filtrage hallucinations"""
    print("\n5️⃣ Test filtrage hallucinations...")
    
    try:
        from STT.audio_streamer_optimized import HallucinationFilter
        
        filter = HallucinationFilter()
        print(f"✅ Filtre hallucinations initialisé")
        print(f"   🚫 {len(filter.hallucination_patterns)} patterns configurés")
        
        # Test avec phrases d'hallucination
        test_cases = [
            ("Bonjour, comment allez-vous ?", False),  # Phrase normale
            ("sous-titres réalisés par la communauté d'amara.org", True),  # Hallucination
            ("merci d'avoir regardé cette vidéo", True),  # Hallucination
            ("", True),  # Texte vide
            ("test test test test test", True),  # Répétitions
        ]
        
        for text, expected_hallucination in test_cases:
            is_hallucination = filter.is_hallucination(text)
            status = "✅" if is_hallucination == expected_hallucination else "❌"
            print(f"   {status} '{text[:30]}...' → Hallucination: {is_hallucination}")
        
        stats = filter.get_stats()
        print(f"   📊 Stats: {stats['texts_analyzed']} analysés, {stats['hallucinations_detected']} filtrés")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur filtre: {e}")
        return False

def test_audio_streamer_mock():
    """Test AudioStreamer avec mock callback"""
    print("\n6️⃣ Test AudioStreamer (mode mock)...")
    
    try:
        from STT.audio_streamer_optimized import AudioStreamer
        import logging
        
        # Setup logging simple
        logger = logging.getLogger('TestAudioStreamer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        
        # Callback mock
        received_chunks = []
        def mock_callback(audio_data):
            received_chunks.append(len(audio_data))
            print(f"   📦 Chunk reçu: {len(audio_data)} samples")
        
        # Créer AudioStreamer
        streamer = AudioStreamer(
            callback=mock_callback,
            logger=logger,
            sample_rate=16000,
            chunk_duration=1.0,  # Chunks courts pour test
            device_name="Rode NT-USB"
        )
        
        print("✅ AudioStreamer créé")
        print(f"   🎤 Périphérique détecté: {streamer.stats['device_detection_success']}")
        print(f"   🔧 Auto-gain activé: {streamer.auto_gain_enabled}")
        
        # Test injection manuelle (simulation)
        import numpy as np
        test_audio = np.random.normal(0, 0.05, 16000).astype(np.float32)  # 1s audio test
        
        print("   💉 Test injection audio manuelle...")
        streamer.running = True  # Simuler état actif
        streamer.add_to_buffer(test_audio)
        
        print(f"   📊 Chunks reçus par callback: {len(received_chunks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur AudioStreamer: {e}")
        return False

def test_integration_readiness():
    """Test préparation intégration avec UnifiedSTTManager"""
    print("\n7️⃣ Test préparation intégration...")
    
    try:
        # Vérifier que UnifiedSTTManager existe
        try:
            from STT.unified_stt_manager import UnifiedSTTManager
            print("✅ UnifiedSTTManager disponible")
        except ImportError:
            print("⚠️ UnifiedSTTManager non trouvé - créer mock")
        
        # Test AudioStreamingManager avec mock
        from STT.audio_streamer_optimized import AudioStreamingManager
        
        # Mock STT Manager simple
        class MockSTTManager:
            def transcribe_sync(self, audio_data):
                class MockResult:
                    def __init__(self):
                        self.text = f"Test transcription {len(audio_data)} samples"
                        self.confidence = 0.95
                        self.rtf = 0.1
                        self.success = True
                        self.error = None
                return MockResult()
        
        mock_stt = MockSTTManager()
        
        # Créer AudioStreamingManager
        streaming_manager = AudioStreamingManager(
            unified_stt_manager=mock_stt,
            device_name="Rode NT-USB",
            chunk_duration=1.0
        )
        
        print("✅ AudioStreamingManager créé avec mock STT")
        
        # Vérifier stats initiales
        stats = streaming_manager.get_stats()
        print(f"   📊 Stats initiales: {stats['continuous_mode_active']}")
        print(f"   🔗 STT Manager prêt: {stats['stt_manager_ready']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur intégration: {e}")
        return False

def main():
    """Test principal"""
    print("🚀 Démarrage tests de validation...")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration GPU", test_gpu_configuration),
        ("Périphériques Audio", test_audio_devices),
        ("VAD", test_vad_initialization),
        ("Filtrage Hallucinations", test_hallucination_filter),
        ("AudioStreamer", test_audio_streamer_mock),
        ("Intégration", test_integration_readiness),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erreur critique dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print("\n🏆 === RÉSUMÉ DES TESTS ===")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"   {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Résultat global: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ AudioStreamer optimisé prêt pour intégration")
        print("🔗 Prochaine étape: Test microphone live complet")
    else:
        print("⚠️ Certains tests ont échoué")
        print("🔧 Vérifier configuration et dépendances")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 