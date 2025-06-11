import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import json

# Since the module sets environment variables on import, 
# we need to patch os.environ BEFORE importing the module.
with patch('os.environ', return_value={}) as mock_env:
    from TTS.tts_handler_piper_fixed import TTSHandlerPiperFixed

@pytest.fixture
def mock_tts_config():
    """Provides a mock configuration for the TTS Handler."""
    return {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
        'use_gpu': True,
        'sample_rate': 22050,
    }

@pytest.fixture
def mock_model_json_config():
    """Provides the content of a mock model JSON config file."""
    return {
        "audio": {"sample_rate": 22050},
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8
        },
        "phoneme_id_map": {
            "^": [1],
            "$": [2],
            "b": [10], "o": [11], "n": [12], "j": [13], "u": [14], "r": [15],
            " ": [0]
        }
    }

@patch('onnxruntime.InferenceSession')
@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
def test_tts_handler_initialization(mock_json_load, mock_file_open, mock_inference_session, mock_tts_config, mock_model_json_config):
    """Tests correct initialization of the TTSHandlerPiperFixed."""
    mock_json_load.return_value = mock_model_json_config
    
    # Mock the ONNX session
    mock_session_instance = MagicMock()
    mock_session_instance.get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    mock_inference_session.return_value = mock_session_instance
    
    handler = TTSHandlerPiperFixed(mock_tts_config)

    # Check config loading
    mock_file_open.assert_called_with(mock_tts_config['config_path'], 'r', encoding='utf-8')
    mock_json_load.assert_called_once()
    assert handler.sample_rate == mock_model_json_config['audio']['sample_rate']
    
    # Check ONNX model loading
    mock_inference_session.assert_called_once()
    assert 'CUDAExecutionProvider' in handler.session.get_providers()
    assert handler.session is not None

def test_text_to_phonemes_simple(mock_model_json_config):
    """Tests the simplified text_to_phonemes conversion."""
    # We can test this method without a full handler instance
    handler = MagicMock() # Mock the handler
    handler.phoneme_id_map = mock_model_json_config['phoneme_id_map']
    handler.text_to_phonemes_simple = TTSHandlerPiperFixed.text_to_phonemes_simple.__get__(handler)

    text = "bonjour"
    expected_ids = [1, 10, 11, 12, 13, 11, 14, 15, 2] # ^, b, o, n, j, o, u, r, $ -> 'o' is reused
    
    phoneme_ids = handler.text_to_phonemes_simple(text)
    
    assert phoneme_ids == expected_ids

@patch('onnxruntime.InferenceSession')
@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
def test_synthesize_successful(mock_json_load, mock_file_open, mock_inference_session, mock_tts_config, mock_model_json_config):
    """Tests a successful synthesis call."""
    mock_json_load.return_value = mock_model_json_config
    
    # Mock the ONNX session to return fake audio data
    mock_session_instance = MagicMock()
    fake_audio_output = np.random.randn(1, 1000).astype(np.float32)
    mock_session_instance.run.return_value = [fake_audio_output]
    mock_inference_session.return_value = mock_session_instance

    handler = TTSHandlerPiperFixed(mock_tts_config)
    audio_data = handler.synthesize("test")
    
    mock_session_instance.run.assert_called_once()
    assert audio_data.dtype == np.int16
    assert audio_data.shape == (1000,)

@patch('sounddevice.play')
@patch('sounddevice.wait')
@patch('TTS.tts_handler_piper_fixed.TTSHandlerPiperFixed.synthesize')
def test_speak_function(mock_synthesize, mock_sd_wait, mock_sd_play, mock_tts_config):
    """Tests that the speak method calls synthesize and plays the audio."""
    # We don't need a full init for this test, just an instance.
    handler = TTSHandlerPiperFixed.__new__(TTSHandlerPiperFixed)
    handler.config = mock_tts_config
    handler.sample_rate = 22050

    fake_audio = np.random.randint(-1000, 1000, size=1000, dtype=np.int16)
    mock_synthesize.return_value = fake_audio
    
    text = "lire ce texte"
    result_audio = handler.speak(text)
    
    mock_synthesize.assert_called_once_with(text)
    mock_sd_play.assert_called_once_with(fake_audio, samplerate=handler.sample_rate)
    mock_sd_wait.assert_called_once()
    assert np.array_equal(result_audio, fake_audio) 