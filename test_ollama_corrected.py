#!/usr/bin/env python3
"""
Test rapide Ollama après correction
"""

import asyncio
import sys
import os
import pathlib

# Setup portable
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Configuration GPU RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

async def test_corrected_llm():
    """Test du LLM Manager corrigé"""
    try:
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        config = {
            'model': 'nous-hermes-2-mistral-7b-dpo:latest',
            'use_ollama': True,
            'timeout': 30.0
        }
        
        print("🧪 Test LLM Manager corrigé...")
        llm_manager = EnhancedLLMManager(config)
        await llm_manager.initialize()
        
        # Test simple
        response = await llm_manager.generate_response("Quelle est la capitale de la France ?")
        
        if response and response.strip():
            print(f"✅ Test réussi: {response}")
            return True
        else:
            print("❌ Test échoué: Pas de réponse")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_corrected_llm())
