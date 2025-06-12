#!/usr/bin/env python3
"""
Tests de validation format audio - SuperWhisper V6 TTS
Vérifie que tous les backends produisent des fichiers WAV valides
"""

import os
import sys
import pytest
import asyncio
import yaml
from pathlib import Path

# Configuration GPU RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Import du système TTS
sys.path.append(str(Path(__file__).parent.parent))
from TTS.tts_manager import UnifiedTTSManager, TTSBackendType
from TTS.utils_audio import is_valid_wav, get_wav_info

@pytest.fixture
def config():
    """Configuration de test pour le TTS Manager"""
    config_path = Path(__file__).parent.parent / "config" / "tts.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@pytest.fixture
async def tts_manager(config):
    """Instance du TTS Manager pour les tests"""
    manager = UnifiedTTSManager(config)
    yield manager
    await manager.cleanup()

@pytest.mark.asyncio
async def test_all_backends_produce_valid_wav(tts_manager):
    """Test que tous les backends produisent un format WAV valide"""
    test_text = "Test de validation format audio"
    
    for backend_type, handler in tts_manager.handlers.items():
        print(f"\n🧪 Test backend: {backend_type.value}")
        
        try:
            # Synthèse directe via le handler
            audio_data = await handler.synthesize(test_text)
            
            # Validation format WAV
            assert audio_data, f"Backend {backend_type.value} a retourné des données vides"
            assert is_valid_wav(audio_data), f"Backend {backend_type.value} ne produit pas un WAV valide"
            
            # Informations détaillées
            wav_info = get_wav_info(audio_data)
            print(f"✅ {backend_type.value}: WAV valide - {wav_info}")
            
            # Validations spécifiques
            assert wav_info.get('channels', 0) > 0, f"Nombre de canaux invalide pour {backend_type.value}"
            assert wav_info.get('framerate', 0) > 0, f"Fréquence d'échantillonnage invalide pour {backend_type.value}"
            assert wav_info.get('size_bytes', 0) > 100, f"Fichier audio trop petit pour {backend_type.value}"
            
        except Exception as e:
            pytest.fail(f"Erreur backend {backend_type.value}: {e}")

@pytest.mark.asyncio
async def test_unified_manager_produces_valid_wav(tts_manager):
    """Test que le manager unifié produit un format WAV valide"""
    test_text = "Test manager unifié format audio"
    
    result = await tts_manager.synthesize(test_text)
    
    assert result.success, f"Synthèse échouée: {result.error}"
    assert result.audio_data, "Aucune donnée audio retournée"
    assert is_valid_wav(result.audio_data), "Le manager unifié ne produit pas un WAV valide"
    
    wav_info = get_wav_info(result.audio_data)
    print(f"✅ Manager unifié: WAV valide - {wav_info}")
    
    # Validation que c'est un fichier audio utilisable
    assert wav_info.get('duration_ms', 0) > 100, "Durée audio trop courte"
    assert wav_info.get('size_bytes', 0) > 1000, "Fichier audio trop petit"

@pytest.mark.asyncio
async def test_cache_preserves_wav_format(tts_manager):
    """Test que le cache préserve le format WAV"""
    test_text = "Test cache format WAV"
    
    # Premier appel (mise en cache)
    result1 = await tts_manager.synthesize(test_text, reuse_cache=True)
    assert result1.success
    assert is_valid_wav(result1.audio_data)
    
    # Deuxième appel (depuis le cache)
    result2 = await tts_manager.synthesize(test_text, reuse_cache=True)
    assert result2.success
    assert result2.backend_used == "cache"
    assert is_valid_wav(result2.audio_data)
    
    # Vérification que les données sont identiques
    assert result1.audio_data == result2.audio_data, "Le cache a altéré les données audio"
    
    print("✅ Cache: Format WAV préservé")

@pytest.mark.asyncio
async def test_fallback_chain_wav_format(tts_manager):
    """Test que la chaîne de fallback produit toujours un format WAV valide"""
    test_text = "Test chaîne fallback"
    
    # Forcer l'échec des premiers backends en les désactivant temporairement
    original_handlers = tts_manager.handlers.copy()
    
    # Test avec seulement SAPI (devrait fonctionner)
    tts_manager.handlers = {
        TTSBackendType.SAPI_FRENCH: original_handlers[TTSBackendType.SAPI_FRENCH],
        TTSBackendType.SILENT_EMERGENCY: original_handlers[TTSBackendType.SILENT_EMERGENCY]
    }
    
    result = await tts_manager.synthesize(test_text)
    assert result.success
    assert is_valid_wav(result.audio_data)
    print(f"✅ Fallback SAPI: WAV valide - Backend: {result.backend_used}")
    
    # Test avec seulement Emergency (silence)
    tts_manager.handlers = {
        TTSBackendType.SILENT_EMERGENCY: original_handlers[TTSBackendType.SILENT_EMERGENCY]
    }
    
    result = await tts_manager.synthesize(test_text)
    assert result.success
    assert is_valid_wav(result.audio_data)
    print(f"✅ Fallback Emergency: WAV valide - Backend: {result.backend_used}")
    
    # Restauration
    tts_manager.handlers = original_handlers

if __name__ == "__main__":
    # Exécution directe pour tests rapides
    import asyncio
    
    async def run_quick_test():
        config_path = Path(__file__).parent.parent / "config" / "tts.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        manager = UnifiedTTSManager(config)
        
        print("🧪 Test rapide validation format WAV...")
        result = await manager.synthesize("Test rapide format audio")
        
        if result.success and is_valid_wav(result.audio_data):
            wav_info = get_wav_info(result.audio_data)
            print(f"✅ Test réussi: {result.backend_used} - {wav_info}")
        else:
            print(f"❌ Test échoué: {result.error}")
        
        await manager.cleanup()
    
    asyncio.run(run_quick_test()) 