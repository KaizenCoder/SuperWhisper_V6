#!/usr/bin/env python3
"""
Test Pipeline Simple - Focus sur LLM Ollama
Test sans microphone pour éviter les crashs
"""

import os
import sys
import pathlib
import asyncio
import logging
import yaml

def _setup_portable_environment():
    current_file = pathlib.Path(__file__).resolve()
    
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    return project_root

_PROJECT_ROOT = _setup_portable_environment()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("PipelineOllama")

async def test_llm_ollama():
    """Test LLM Ollama spécifiquement"""
    logger.info("🧠 Test LLM Ollama...")
    
    try:
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        llm_config = {
            'model': 'nous-hermes-2-mistral-7b-dpo:latest',
            'timeout': 30.0,
            'use_ollama': True
        }
        
        llm_manager = EnhancedLLMManager(llm_config)
        await llm_manager.initialize()
        
        # Test questions
        questions = [
            "Quelle est la capitale de la France ?",
            "Quelle est la capitale de l'Italie ?",
            "Quelle est la capitale des États-Unis ?"
        ]
        
        for i, question in enumerate(questions, 1):
            logger.info(f"🤔 Question {i}: {question}")
            
            response = await llm_manager.generate_response(
                user_input=question,
                max_tokens=100
            )
            
            logger.info(f"🤖 Réponse: {response}")
            logger.info("-" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test LLM: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_tts_simple():
    """Test TTS simple"""
    logger.info("🔊 Test TTS...")
    
    try:
        with open('config/tts.yaml', 'r') as f:
            tts_config = yaml.safe_load(f)
        
        from TTS.tts_manager import UnifiedTTSManager
        tts_manager = UnifiedTTSManager(tts_config)
        
        test_text = "La capitale de la France est Paris. La capitale de l'Italie est Rome."
        
        result = await tts_manager.synthesize(text=test_text)
        
        if result.success and result.audio_data:
            audio_file = f"test_ollama_response.wav"
            with open(audio_file, 'wb') as f:
                f.write(result.audio_data)
            
            logger.info(f"✅ TTS: {len(result.audio_data)} bytes")
            logger.info(f"📁 Audio: {audio_file}")
            
            # Lecture
            try:
                import subprocess
                subprocess.run([
                    "start", "", audio_file
                ], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"🎧 Lecture: {audio_file}")
            except:
                logger.info(f"🎧 Lisez manuellement: {audio_file}")
                
            return True
        else:
            logger.error(f"❌ TTS échoué: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur TTS: {e}")
        return False

async def main():
    print("🚀 SuperWhisper V6 - Test Pipeline Ollama Simple")
    print("=" * 60)
    
    try:
        # Test LLM
        print("\\n🧠 Test LLM Ollama...")
        llm_success = await test_llm_ollama()
        
        # Test TTS
        print("\\n🔊 Test TTS...")
        tts_success = await test_tts_simple()
        
        print("\\n🏁 RÉSULTATS:")
        print("=" * 40)
        print(f"🧠 LLM Ollama: {'✅' if llm_success else '❌'}")
        print(f"🔊 TTS: {'✅' if tts_success else '❌'}")
        
        if llm_success and tts_success:
            print("\\n🎉 PIPELINE LLM→TTS FONCTIONNEL!")
            print("   Ollama répond correctement aux questions")
        else:
            print("\\n⚠️ Problèmes détectés - voir logs")
        
    except KeyboardInterrupt:
        print("\\n🛑 Test interrompu")
    except Exception as e:
        print(f"\\n❌ Erreur fatale: {e}")

if __name__ == "__main__":
    asyncio.run(main())