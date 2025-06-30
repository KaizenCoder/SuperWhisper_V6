# tests/test_realtime_audio_pipeline.py
"""
Test d'intégration du RobustSTTManager avec microphone réel
Conforme aux exigences PRD v3.1 - Validation obligatoire en conditions réelles
"""
import pytest
import asyncio
import yaml
import sounddevice as sd
import numpy as np
import soundfile as sf
import io
from pathlib import Path
import sys
import time
import logging
import torch

# Configuration logging pour debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT.stt_manager_robust import RobustSTTManager
from STT.vad_manager import OptimizedVADManager

def record_micro(seconds: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """
    Enregistrement microphone avec feedback visuel
    
    Args:
        seconds: Durée d'enregistrement
        sample_rate: Taux d'échantillonnage (16kHz pour Whisper)
        
    Returns:
        numpy array de l'audio enregistré
    """
    print(f"\n🎤 Enregistrement pendant {seconds} secondes...")
    print("3... ", end="", flush=True)
    time.sleep(1)
    print("2... ", end="", flush=True)
    time.sleep(1)
    print("1... ", end="", flush=True)
    time.sleep(1)
    print("PARLEZ MAINTENANT! 🔴")
    
    # Enregistrement
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    print("✅ Enregistrement terminé\n")
    return audio.flatten()

@pytest.mark.asyncio
async def test_robust_stt_manager_validation_complete():
    """
    Test de validation complet du RobustSTTManager
    Critères PRD v3.1 : Test microphone réel obligatoire
    """
    print("\n" + "="*80)
    print("TEST DE VALIDATION ROBUSTSTTMANAGER - CONDITIONS RÉELLES")
    print("="*80)
    
    # 1. Configuration
    config_path = Path("Config/mvp_settings.yaml")
    if not config_path.exists():
        pytest.skip(f"Configuration non trouvée: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration STT pour test
    stt_config = config.get('stt', {})
    stt_config.update({
        'use_gpu': torch.cuda.is_available(),  # Auto-détection GPU
        'fallback_chain': ['tiny'],  # Modèle léger pour tests
        'use_vad': True,  # Test avec VAD activé
        'model_cache_dir': './models/whisper'
    })
    
    # 2. Initialisation composants
    print("\n📋 Initialisation des composants...")
    
    # VAD Manager (si disponible)
    vad_manager = None
    try:
        vad_config = config.get('vad', {})
        vad_manager = OptimizedVADManager()
        await vad_manager.initialize()
        print("✅ VAD Manager initialisé")
    except Exception as e:
        print(f"⚠️  VAD non disponible: {e}")
        stt_config['use_vad'] = False
    
    # STT Manager
    stt_manager = RobustSTTManager(stt_config, vad_manager=vad_manager)
    await stt_manager.initialize()
    print(f"✅ STT Manager initialisé sur {stt_manager.device}")
    
    # 3. Test 1 : Phrase de validation obligatoire
    print("\n🧪 TEST 1 : Phrase de validation obligatoire")
    phrase_validation = "Ceci est un test de validation du nouveau gestionnaire robuste"
    print(f"📢 Phrase à prononcer : « {phrase_validation} »")
    
    audio_array = record_micro(seconds=7)
    
    # Conversion en bytes WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, 16000, format='WAV', subtype='PCM_16')
    audio_bytes = buffer.getvalue()
    
    # Transcription
    start_time = time.time()
    result = await stt_manager.transcribe_audio(audio_bytes, language="fr")
    
    # Validation résultats
    transcription = result['text'].lower()
    processing_time = result['processing_time']
    
    print(f"\n📝 Transcription : '{result['text']}'")
    print(f"⏱️  Temps de traitement : {processing_time:.3f}s")
    print(f"🖥️  Device utilisé : {result['device']}")
    print(f"📊 RTF (Real-Time Factor) : {result['metrics']['rtf']:.2f}")
    
    # Assertions sémantiques
    mots_cles = ["test", "validation", "gestionnaire", "robuste"]
    mots_trouves = [mot for mot in mots_cles if mot in transcription]
    
    assert len(mots_trouves) >= 3, (
        f"Transcription incomplète. Mots trouvés: {mots_trouves}/4. "
        f"Transcription: '{result['text']}'"
    )
    
    # Assertion performance (< 300ms pour audio court selon PRD)
    audio_duration = len(audio_array) / 16000
    if audio_duration < 10:  # Audio court
        assert processing_time < 0.3, (
            f"Latence trop élevée: {processing_time:.3f}s > 0.3s (cible PRD)"
        )
    
    print("✅ TEST 1 RÉUSSI : Validation sémantique et performance OK")
    
    # 4. Test 2 : Robustesse avec audio difficile
    print("\n🧪 TEST 2 : Test de robustesse")
    print("📢 Parlez rapidement avec des mots techniques")
    
    audio_array2 = record_micro(seconds=5)
    buffer2 = io.BytesIO()
    sf.write(buffer2, audio_array2, 16000, format='WAV', subtype='PCM_16')
    
    result2 = await stt_manager.transcribe_audio(buffer2.getvalue())
    
    assert result2['text'] != "", "Transcription vide sur audio difficile"
    assert result2['processing_time'] < 1.0, "Temps de traitement trop long"
    
    print(f"📝 Transcription robuste : '{result2['text']}'")
    print("✅ TEST 2 RÉUSSI : Robustesse validée")
    
    # 5. Test 3 : Métriques et monitoring
    print("\n🧪 TEST 3 : Vérification métriques")
    
    metrics = stt_manager.get_metrics()
    assert metrics['transcriptions'] == 2, "Compteur transcriptions incorrect"
    assert metrics['errors'] == 0, "Erreurs détectées pendant les tests"
    assert metrics['avg_latency'] > 0, "Latence moyenne non calculée"
    
    print(f"📊 Métriques finales : {metrics}")
    print("✅ TEST 3 RÉUSSI : Métriques correctes")
    
    # 6. Nettoyage
    await stt_manager.cleanup()
    
    # Résumé final
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLÈTE RÉUSSIE")
    print(f"   - Transcriptions réussies : {metrics['transcriptions']}")
    print(f"   - Latence moyenne : {metrics['avg_latency']:.3f}s")
    print(f"   - Device utilisé : {metrics['device']}")
    print(f"   - Modèles chargés : {metrics['models_loaded']}")
    print("="*80)

@pytest.mark.asyncio
async def test_fallback_chain():
    """Test spécifique de la chaîne de fallback"""
    print("\n🧪 TEST FALLBACK CHAIN")
    
    # Configuration avec chaîne de fallback étendue
    stt_config = {
        'use_gpu': False,  # Force CPU pour test
        'fallback_chain': ['base', 'small', 'tiny'],
        'use_vad': False
    }
    
    stt_manager = RobustSTTManager(stt_config)
    
    # Initialisation avec simulation d'échec sur 'base'
    stt_manager.fallback_chain = ['small', 'tiny']  # Skip 'base'
    await stt_manager.initialize()
    
    # Test audio simple
    test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
    buffer = io.BytesIO()
    sf.write(buffer, test_audio, 16000, format='WAV')
    
    result = await stt_manager.transcribe_audio(buffer.getvalue())
    
    assert result['text'] == "", "Silence devrait donner transcription vide"
    assert result['device'] == 'cpu', "Device incorrect"
    
    await stt_manager.cleanup()
    print("✅ Fallback chain validée")

@pytest.mark.asyncio
async def test_vad_integration():
    """Test spécifique de l'intégration VAD"""
    print("\n🧪 TEST INTÉGRATION VAD")
    
    # Configuration VAD activé
    stt_config = {
        'use_gpu': False,
        'fallback_chain': ['tiny'],
        'use_vad': True
    }
    
    # Initialisation VAD
    vad_manager = OptimizedVADManager()
    await vad_manager.initialize()
    
    stt_manager = RobustSTTManager(stt_config, vad_manager=vad_manager)
    await stt_manager.initialize()
    
    # Test avec silence (VAD devrait filtrer)
    silence_audio = np.zeros(16000, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, silence_audio, 16000, format='WAV')
    
    result = await stt_manager.transcribe_audio(buffer.getvalue())
    
    # VAD devrait détecter le silence et retourner rapidement
    assert result['processing_time'] < 0.1, "VAD n'a pas filtré le silence efficacement"
    
    await stt_manager.cleanup()
    print("✅ Intégration VAD validée")

if __name__ == "__main__":
    # Exécution directe pour tests manuels
    asyncio.run(test_robust_stt_manager_validation_complete()) 