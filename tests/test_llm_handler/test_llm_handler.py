import pytest
from unittest.mock import patch, MagicMock
from LLM.llm_handler import LLMHandler

@pytest.fixture
def mock_config():
    """Fixture to provide a mock configuration for the LLM Handler."""
    return {
        'model_path': '/fake/path/to/model.gguf',
        'n_gpu_layers': 32,
        'gpu_device_index': 0,  # RTX 3090 (CUDA:0) - NE PAS UTILISER 1 (RTX 5060)
    }

@patch('LLM.llm_handler.Llama')
def test_llm_handler_initialization(mock_llama, mock_config):
    """Tests that the LLMHandler initializes the Llama model correctly."""
    handler = LLMHandler(mock_config)
    
    mock_llama.assert_called_once_with(
        model_path=mock_config['model_path'],
        n_gpu_layers=mock_config['n_gpu_layers'],
        main_gpu=mock_config['gpu_device_index'],
        verbose=False
    )
    assert handler.llm == mock_llama.return_value

@patch('LLM.llm_handler.Llama')
def test_get_response_success(mock_llama_class, mock_config):
    """Tests the get_response method for a successful interaction."""
    # Setup the mock for the Llama instance
    mock_llm_instance = MagicMock()
    mock_llm_instance.return_value = {
        'choices': [{'text': ' Voici une réponse de test.'}]
    }
    mock_llama_class.return_value = mock_llm_instance

    handler = LLMHandler(mock_config)
    
    prompt = "Quelle est la question ?"
    response = handler.get_response(prompt)
    
    # Verify that the mock was called correctly
    mock_llm_instance.assert_called_once_with(
        f"Q: {prompt} A: ", 
        max_tokens=100, 
        stop=["Q:", "\n"]
    )
    
    # Verify the response is cleaned up correctly
    assert response == "Voici une réponse de test."

@patch('LLM.llm_handler.Llama')
def test_get_response_empty_choice(mock_llama_class, mock_config):
    """Tests how get_response handles an empty or malformed response from the model."""
    mock_llm_instance = MagicMock()
    # Simulate a response with no text
    mock_llm_instance.return_value = {
        'choices': [{'text': ''}]
    }
    mock_llama_class.return_value = mock_llm_instance
    
    handler = LLMHandler(mock_config)
    response = handler.get_response("Un prompt")
    
    assert response == ""

@patch('LLM.llm_handler.Llama')
def test_get_response_no_choices(mock_llama_class, mock_config):
    """Tests how get_response handles a response with no 'choices' array."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.return_value = {'choices': []} # No choices
    mock_llama_class.return_value = mock_llm_instance
    
    handler = LLMHandler(mock_config)
    
    # It should raise an IndexError because the code tries to access choices[0]
    with pytest.raises(IndexError):
        handler.get_response("Un autre prompt") 