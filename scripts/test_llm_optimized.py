#!/usr/bin/env python3
"""
🚀 TEST LLM OPTIMISÉ - SUPERWHISPER V6
====================================
Test LLM avec optimisations pour réduire latence < 400ms

OPTIMISATIONS:
- Réponses courtes (max 30 tokens)
- Temperature basse (0.3)
- Modèle le plus rapide (qwen2.5-coder:1.5b)
- Système prompt optimisé

Usage: python scripts/test_llm_optimized.py

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

import asyncio
import time
import logging
import httpx

# Configuration GPU RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("llm_optimized")

class OptimizedLLMTester:
    """Testeur LLM optimisé pour latence minimale"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.timeout = 15
        self.best_model = "qwen2.5-coder:1.5b"  # Modèle le plus rapide identifié
        
        # Prompts optimisés pour réponses courtes
        self.test_prompts = [
            "Bonjour !",
            "Météo ?", 
            "Heure ?",
            "Merci !",
            "Au revoir"
        ]
    
    async def test_optimized_inference(self, prompt: str) -> dict:
        """Test inférence optimisée"""
        start_time = time.time()
        
        try:
            # Configuration optimisée pour vitesse
            payload = {
                'model': self.best_model,
                'prompt': f"Réponds en 1-3 mots maximum en français: {prompt}",
                'stream': False,
                'options': {
                    'temperature': 0.1,    # Très déterministe 
                    'num_predict': 15,     # Maximum 15 tokens
                    'top_p': 0.8,         # Focus sur tokens probables
                    'top_k': 20,          # Limitation vocabulaire
                    'repeat_penalty': 1.1,# Éviter répétitions
                    'stop': ['.', '!', '?', '\n']  # Arrêt précoce
                }
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                response_text = data.get('response', '').strip()
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    'success': True,
                    'latency_ms': latency_ms,
                    'response_text': response_text,
                    'tokens': len(response_text.split())
                }
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                'success': False,
                'latency_ms': latency_ms,
                'error': str(e)
            }
    
    async def run_optimization_test(self):
        """Test optimisation complète"""
        logger.info(f"🚀 TEST OPTIMISATION LLM: {self.best_model}")
        logger.info("🎯 Objectif: Réduire latence via réponses courtes")
        
        latencies = []
        responses = []
        
        for i, prompt in enumerate(self.test_prompts):
            logger.info(f"Test {i+1}/5: '{prompt}'")
            
            result = await self.test_optimized_inference(prompt)
            
            if result['success']:
                latencies.append(result['latency_ms'])
                responses.append(result['response_text'])
                
                logger.info(f"   ✅ {result['latency_ms']:.1f}ms - '{result['response_text']}' ({result['tokens']} tokens)")
            else:
                logger.warning(f"   ❌ Échec: {result.get('error', 'Erreur inconnue')}")
        
        # Analyse résultats
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print("\n" + "="*60)
            print("🎯 RÉSULTATS OPTIMISATION LLM")
            print("="*60)
            print(f"⚡ Latence moyenne: {avg_latency:.1f}ms")
            print(f"🚀 Latence minimum: {min_latency:.1f}ms")
            print(f"⏱️ Latence maximum: {max_latency:.1f}ms")
            print(f"🎯 Objectif < 400ms: {'✅ ATTEINT' if avg_latency < 400 else '❌ DÉPASSÉ'}")
            
            # Projection pipeline complet
            stt_latency = 833  # Validé
            tts_latency = 975  # Validé
            total_pipeline = stt_latency + avg_latency + tts_latency
            
            print(f"\n📊 PROJECTION PIPELINE COMPLET:")
            print(f"   STT: {stt_latency}ms")
            print(f"   LLM: {avg_latency:.1f}ms (optimisé)")
            print(f"   TTS: {tts_latency}ms")
            print(f"   TOTAL: {total_pipeline:.1f}ms")
            print(f"   Objectif < 1200ms: {'✅ ATTEINT' if total_pipeline < 1200 else '❌ DÉPASSÉ'}")
            
            if total_pipeline < 2500:
                print(f"   Objectif < 2500ms: ✅ RÉALISTE")
            
            print("="*60)
            
            # Exemples réponses
            if responses:
                print(f"\n💬 EXEMPLES RÉPONSES OPTIMISÉES:")
                for prompt, response in zip(self.test_prompts, responses):
                    print(f"   '{prompt}' → '{response}'")
            
            return avg_latency < 400
        
        return False

async def main():
    """Test principal optimisation LLM"""
    try:
        tester = OptimizedLLMTester()
        success = await tester.run_optimization_test()
        
        if success:
            print("\n🎊 OPTIMISATION RÉUSSIE - Modèle prêt pour pipeline")
            return 0
        else:
            print("\n⚠️ OPTIMISATION LIMITÉE - Objectifs à réviser")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Erreur test optimisation: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 