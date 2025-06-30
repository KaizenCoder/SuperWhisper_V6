#!/usr/bin/env python3
"""
⚡ OPTIMISATION NOUS-HERMES-2-MISTRAL-7B-DPO - SUPERWHISPER V6
============================================================
Script d'optimisation pour réduire latence < 600ms

PROBLÈME IDENTIFIÉ:
- Latence actuelle: 1035ms (objectif: < 600ms)
- Pipeline total: 2.84s (objectif: < 2.5s)

STRATÉGIES D'OPTIMISATION:
1. Réduction tokens de sortie (50 → 25)
2. Temperature plus basse (0.7 → 0.3)
3. Paramètres Ollama optimisés
4. Test configurations ultra-rapides

Usage: python scripts/optimize_nous_hermes.py

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
import statistics
from typing import Dict, Any, List
import httpx

# Configuration GPU RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nous_hermes_optimizer")

class NousHermesOptimizer:
    """Optimiseur de latence pour Nous-Hermes-2-Mistral-7B-DPO"""
    
    def __init__(self):
        self.model_name = "nous-hermes-2-mistral-7b-dpo:latest"
        self.ollama_url = "http://localhost:11434"
        self.target_latency = 600  # ms
        
        # Prompts de test optimisés (plus courts)
        self.test_prompts = [
            "Bonjour !",
            "Quelle heure ?",
            "Merci !",
            "Ça va ?",
            "Au revoir"
        ]
        
        # Configurations optimisées pour vitesse
        self.optimization_configs = [
            {
                "name": "ultra_fast",
                "temperature": 0.1,
                "max_tokens": 15,
                "top_p": 0.7,
                "top_k": 20,
                "repeat_penalty": 1.0
            },
            {
                "name": "very_fast", 
                "temperature": 0.3,
                "max_tokens": 20,
                "top_p": 0.8,
                "top_k": 30,
                "repeat_penalty": 1.05
            },
            {
                "name": "fast_balanced",
                "temperature": 0.5,
                "max_tokens": 25,
                "top_p": 0.85,
                "top_k": 40,
                "repeat_penalty": 1.1
            },
            {
                "name": "minimal_tokens",
                "temperature": 0.2,
                "max_tokens": 10,
                "top_p": 0.75,
                "top_k": 25,
                "repeat_penalty": 1.0
            }
        ]
    
    async def test_optimization_config(self, config: Dict):
        """Tester une configuration d'optimisation"""
        logger.info(f"🔧 Test configuration '{config['name']}'...")
        
        results = []
        
        for prompt in self.test_prompts:
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config["temperature"],
                        "num_predict": config["max_tokens"],
                        "top_p": config["top_p"],
                        "top_k": config["top_k"],
                        "repeat_penalty": config["repeat_penalty"]
                    }
                }
                
                start_time = time.time()
                
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "").strip()
                    
                    # Évaluation qualité basique
                    quality = self.evaluate_basic_quality(response_text, prompt)
                    
                    results.append({
                        "prompt": prompt,
                        "latency": latency,
                        "response": response_text,
                        "quality": quality,
                        "success": True
                    })
                    
                    logger.info(f"✅ '{prompt}': {latency:.1f}ms, qualité: {quality:.1f}/10")
                else:
                    logger.error(f"❌ '{prompt}': HTTP {response.status_code}")
                    results.append({"success": False})
                    
            except Exception as e:
                logger.error(f"❌ Erreur '{prompt}': {e}")
                results.append({"success": False})
        
        return results
    
    def evaluate_basic_quality(self, response: str, prompt: str) -> float:
        """Évaluation qualité basique (0-10)"""
        if not response or len(response) < 2:
            return 0.0
        
        score = 5.0
        
        # Vérification longueur appropriée pour réponse courte
        if 5 <= len(response) <= 50:
            score += 2.0
        elif len(response) < 5:
            score -= 2.0
        
        # Vérification français basique
        french_words = ["bonjour", "merci", "bien", "oui", "non", "salut", "ça", "va"]
        if any(word in response.lower() for word in french_words):
            score += 1.5
        
        # Cohérence avec prompt
        if "bonjour" in prompt.lower() and any(word in response.lower() for word in ["bonjour", "salut", "aide"]):
            score += 1.0
        elif "merci" in prompt.lower() and any(word in response.lower() for word in ["rien", "plaisir", "service"]):
            score += 1.0
        elif "heure" in prompt.lower() and any(word in response.lower() for word in ["heure", "temps"]):
            score += 1.0
        
        # Pénalités
        if "je suis prêt" in response.lower():
            score -= 3.0
        
        return max(0.0, min(10.0, score))
    
    async def run_optimization_tests(self):
        """Exécuter tous les tests d'optimisation"""
        logger.info("⚡ DÉMARRAGE OPTIMISATION NOUS-HERMES-2-MISTRAL-7B-DPO")
        
        all_results = {}
        
        for config in self.optimization_configs:
            results = await self.test_optimization_config(config)
            successful_results = [r for r in results if r.get("success", False)]
            
            if successful_results:
                avg_latency = statistics.mean([r["latency"] for r in successful_results])
                avg_quality = statistics.mean([r["quality"] for r in successful_results])
                success_rate = len(successful_results) / len(results) * 100
                
                all_results[config["name"]] = {
                    "config": config,
                    "avg_latency": avg_latency,
                    "avg_quality": avg_quality,
                    "success_rate": success_rate,
                    "results": successful_results
                }
                
                logger.info(f"📊 {config['name']}: {avg_latency:.1f}ms, qualité: {avg_quality:.1f}/10, succès: {success_rate:.1f}%")
        
        return self.analyze_optimization_results(all_results)
    
    def analyze_optimization_results(self, all_results: Dict):
        """Analyser les résultats d'optimisation"""
        if not all_results:
            logger.error("❌ Aucun résultat d'optimisation")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("⚡ ANALYSE OPTIMISATION NOUS-HERMES-2-MISTRAL-7B-DPO")
        logger.info("="*60)
        
        # Trouver la meilleure configuration
        best_config = None
        best_score = 0
        
        for name, data in all_results.items():
            # Score pondéré: latence (60%) + qualité (40%)
            latency_score = max(0, (self.target_latency - data["avg_latency"]) / self.target_latency * 10)
            quality_score = data["avg_quality"]
            weighted_score = (latency_score * 0.6) + (quality_score * 0.4)
            
            logger.info(f"🔧 {name}:")
            logger.info(f"   ⚡ Latence: {data['avg_latency']:.1f}ms")
            logger.info(f"   🎯 Qualité: {data['avg_quality']:.1f}/10")
            logger.info(f"   ✅ Succès: {data['success_rate']:.1f}%")
            logger.info(f"   📊 Score: {weighted_score:.1f}/10")
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_config = (name, data)
        
        if best_config:
            name, data = best_config
            logger.info(f"\n🏆 MEILLEURE CONFIGURATION: {name}")
            logger.info(f"⚡ Latence optimisée: {data['avg_latency']:.1f}ms")
            logger.info(f"🎯 Qualité: {data['avg_quality']:.1f}/10")
            
            # Projection pipeline optimisé
            stt_latency = 833
            tts_latency = 975
            optimized_pipeline = stt_latency + data['avg_latency'] + tts_latency
            
            logger.info(f"\n📊 PIPELINE OPTIMISÉ:")
            logger.info(f"🎤 STT: {stt_latency}ms")
            logger.info(f"🤖 LLM: {data['avg_latency']:.1f}ms")
            logger.info(f"🔊 TTS: {tts_latency}ms")
            logger.info(f"📊 TOTAL: {optimized_pipeline:.1f}ms ({optimized_pipeline/1000:.2f}s)")
            
            # Évaluation objectifs
            latency_ok = data['avg_latency'] <= self.target_latency
            pipeline_ok = optimized_pipeline <= 2500
            quality_ok = data['avg_quality'] >= 6.0  # Seuil réduit pour optimisation vitesse
            
            if latency_ok:
                logger.info("✅ Objectif latence < 600ms: ATTEINT")
            else:
                logger.warning(f"⚠️ Objectif latence < 600ms: DÉPASSÉ (+{data['avg_latency'] - self.target_latency:.1f}ms)")
            
            if pipeline_ok:
                logger.info("✅ Objectif pipeline < 2.5s: ATTEINT")
            else:
                logger.warning(f"⚠️ Objectif pipeline < 2.5s: DÉPASSÉ (+{(optimized_pipeline-2500)/1000:.2f}s)")
            
            if quality_ok:
                logger.info("✅ Qualité acceptable ≥ 6.0/10: ATTEINT")
            else:
                logger.warning("⚠️ Qualité acceptable ≥ 6.0/10: NON ATTEINT")
            
            # Verdict final
            overall_success = latency_ok and quality_ok
            
            if overall_success:
                logger.info("\n🎊 OPTIMISATION RÉUSSIE !")
                logger.info("🚀 Configuration optimale trouvée pour SuperWhisper V6")
                self.save_optimal_config(name, data['config'])
                return True
            else:
                logger.warning("\n⚠️ OPTIMISATION PARTIELLE")
                logger.info("🔧 Compromis vitesse/qualité nécessaire")
                return False
        
        return False
    
    def save_optimal_config(self, config_name: str, config: Dict):
        """Sauvegarder la configuration optimale"""
        try:
            config_path = "docs/OPTIMAL_NOUS_HERMES_CONFIG.md"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("# ⚡ CONFIGURATION OPTIMALE NOUS-HERMES-2-MISTRAL-7B-DPO\n\n")
                f.write(f"## 🏆 Configuration sélectionnée: {config_name}\n\n")
                f.write("### 📋 Paramètres Ollama\n\n")
                f.write("```json\n")
                f.write("{\n")
                f.write(f'  "model": "nous-hermes-2-mistral-7b-dpo:latest",\n')
                f.write(f'  "temperature": {config["temperature"]},\n')
                f.write(f'  "num_predict": {config["max_tokens"]},\n')
                f.write(f'  "top_p": {config["top_p"]},\n')
                f.write(f'  "top_k": {config["top_k"]},\n')
                f.write(f'  "repeat_penalty": {config["repeat_penalty"]}\n')
                f.write("}\n")
                f.write("```\n\n")
                f.write("### 🚀 Utilisation en production\n\n")
                f.write("Cette configuration optimise la latence pour SuperWhisper V6 tout en maintenant une qualité conversationnelle acceptable.\n")
            
            logger.info(f"📄 Configuration optimale sauvegardée: {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde config: {e}")

async def main():
    """Fonction principale d'optimisation"""
    try:
        optimizer = NousHermesOptimizer()
        success = await optimizer.run_optimization_tests()
        
        if success:
            print("\n🎊 OPTIMISATION TERMINÉE AVEC SUCCÈS")
            print("⚡ Configuration optimale trouvée pour SuperWhisper V6")
            return 0
        else:
            print("\n⚠️ OPTIMISATION PARTIELLE")
            print("🔧 Compromis vitesse/qualité identifié")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Erreur optimisation: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 