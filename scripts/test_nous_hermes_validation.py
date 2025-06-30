#!/usr/bin/env python3
"""
🦙 VALIDATION NOUS-HERMES-2-MISTRAL-7B-DPO - SUPERWHISPER V6
===========================================================
Script validation complète pour Nous-Hermes-2-Mistral-7B-DPO

MODÈLE CIBLE:
- Nom: nous-hermes-2-mistral-7b-dpo
- Fichier: Nous-Hermes-2-Mistral-7B-DPO.Q4_K_S.gguf (3.9GB)
- Avantages: DPO-tuned, conversation excellente, français natif

MISSION:
- Validation latence < 600ms pour pipeline < 2.5s
- Tests conversation française approfondie
- Évaluation qualité contextuelle
- Sélection paramètres optimaux

Usage: python scripts/test_nous_hermes_validation.py

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

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Configuration logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nous_hermes_validation")

class NousHermesValidator:
    """Validateur complet pour Nous-Hermes-2-Mistral-7B-DPO"""
    
    def __init__(self):
        self.model_name = "nous-hermes-2-mistral-7b-dpo:latest"
        self.ollama_url = "http://localhost:11434"
        self.target_latency = 600  # ms - objectif pour pipeline < 2.5s
        
        # Prompts de test conversation française
        self.test_prompts = [
            {
                "category": "greeting",
                "prompt": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                "expected_keywords": ["bonjour", "aide", "service"]
            },
            {
                "category": "time_question", 
                "prompt": "Quelle heure est-il maintenant ?",
                "expected_keywords": ["heure", "temps", "maintenant"]
            },
            {
                "category": "gratitude",
                "prompt": "Merci beaucoup pour votre aide précieuse !",
                "expected_keywords": ["merci", "plaisir", "service"]
            },
            {
                "category": "weather",
                "prompt": "Quel temps fait-il dehors ?",
                "expected_keywords": ["temps", "météo", "information"]
            },
            {
                "category": "conversation",
                "prompt": "Racontez-moi une histoire courte et amusante.",
                "expected_keywords": ["histoire", "fois", "alors"]
            },
            {
                "category": "help_request",
                "prompt": "J'ai besoin d'aide pour organiser ma journée.",
                "expected_keywords": ["aide", "organiser", "planifier"]
            },
            {
                "category": "casual_chat",
                "prompt": "Comment ça va de votre côté ?",
                "expected_keywords": ["bien", "merci", "assistant"]
            }
        ]
        
        # Configurations de test
        self.test_configs = [
            {
                "name": "optimal",
                "temperature": 0.7,
                "max_tokens": 50,
                "top_p": 0.9
            },
            {
                "name": "fast",
                "temperature": 0.3,
                "max_tokens": 30,
                "top_p": 0.8
            },
            {
                "name": "creative",
                "temperature": 0.9,
                "max_tokens": 80,
                "top_p": 0.95
            }
        ]
    
    async def validate_model_available(self):
        """Valider que le modèle est disponible dans Ollama"""
        logger.info(f"🔍 Validation disponibilité modèle {self.model_name}...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                
                if response.status_code == 200:
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    
                    if self.model_name in models:
                        logger.info(f"✅ Modèle {self.model_name} disponible")
                        return True
                    else:
                        logger.error(f"❌ Modèle {self.model_name} non trouvé")
                        logger.info(f"📋 Modèles disponibles: {models}")
                        return False
                else:
                    logger.error(f"❌ Erreur Ollama: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur validation: {e}")
            return False
    
    async def test_single_prompt(self, prompt_data: Dict, config: Dict):
        """Tester un prompt avec une configuration donnée"""
        prompt = prompt_data["prompt"]
        category = prompt_data["category"]
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config["temperature"],
                    "num_predict": config["max_tokens"],
                    "top_p": config["top_p"]
                }
            }
            
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Évaluation qualité
                quality_score = self.evaluate_response_quality(
                    response_text, 
                    prompt_data["expected_keywords"],
                    category
                )
                
                return {
                    "success": True,
                    "latency": latency,
                    "response": response_text,
                    "quality_score": quality_score,
                    "category": category,
                    "config": config["name"]
                }
            else:
                logger.error(f"❌ Erreur HTTP {response.status_code}")
                return {"success": False, "latency": 0, "quality_score": 0}
                
        except Exception as e:
            logger.error(f"❌ Erreur test: {e}")
            return {"success": False, "latency": 0, "quality_score": 0}
    
    def evaluate_response_quality(self, response: str, expected_keywords: List[str], category: str) -> float:
        """Évaluer la qualité d'une réponse (0-10)"""
        if not response or len(response) < 5:
            return 0.0
        
        score = 5.0  # Score de base
        
        # Vérification longueur appropriée
        if 10 <= len(response) <= 200:
            score += 1.0
        elif len(response) < 10:
            score -= 2.0
        
        # Vérification français
        french_indicators = ["je", "vous", "est", "pour", "avec", "dans", "sur", "une", "des", "les"]
        french_count = sum(1 for word in french_indicators if word in response.lower())
        score += min(french_count * 0.2, 1.5)
        
        # Vérification mots-clés contextuels
        keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in response.lower())
        if keyword_matches > 0:
            score += keyword_matches * 0.5
        
        # Vérification cohérence conversationnelle
        if category == "greeting" and any(word in response.lower() for word in ["bonjour", "salut", "aide"]):
            score += 1.0
        elif category == "gratitude" and any(word in response.lower() for word in ["plaisir", "rien", "service"]):
            score += 1.0
        elif category == "time_question" and any(word in response.lower() for word in ["heure", "temps", "information"]):
            score += 1.0
        
        # Pénalités
        if "je suis prêt" in response.lower():
            score -= 3.0  # Réponse répétitive
        
        if response.count(".") == 0 and len(response) > 20:
            score -= 0.5  # Manque de ponctuation
        
        return max(0.0, min(10.0, score))
    
    async def run_comprehensive_validation(self):
        """Validation complète du modèle"""
        logger.info("🚀 DÉMARRAGE VALIDATION NOUS-HERMES-2-MISTRAL-7B-DPO")
        
        # 1. Validation disponibilité
        if not await self.validate_model_available():
            return False
        
        all_results = []
        
        # 2. Tests avec différentes configurations
        for config in self.test_configs:
            logger.info(f"🔧 Test configuration '{config['name']}'...")
            
            config_results = []
            
            for prompt_data in self.test_prompts:
                result = await self.test_single_prompt(prompt_data, config)
                
                if result["success"]:
                    config_results.append(result)
                    logger.info(f"✅ {prompt_data['category']}: {result['latency']:.1f}ms, qualité: {result['quality_score']:.1f}/10")
                else:
                    logger.error(f"❌ {prompt_data['category']}: échec")
            
            all_results.extend(config_results)
        
        # 3. Analyse résultats
        return self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict]) -> bool:
        """Analyser les résultats de validation"""
        if not results:
            logger.error("❌ Aucun résultat à analyser")
            return False
        
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            logger.error("❌ Aucun test réussi")
            return False
        
        # Calculs statistiques
        latencies = [r["latency"] for r in successful_results]
        quality_scores = [r["quality_score"] for r in successful_results]
        
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_quality = statistics.mean(quality_scores)
        
        success_rate = len(successful_results) / len(results) * 100
        
        # Rapport final
        logger.info("\n" + "="*60)
        logger.info("📊 RAPPORT VALIDATION NOUS-HERMES-2-MISTRAL-7B-DPO")
        logger.info("="*60)
        logger.info(f"✅ Taux de succès: {success_rate:.1f}% ({len(successful_results)}/{len(results)})")
        logger.info(f"⚡ Latence moyenne: {avg_latency:.1f}ms")
        logger.info(f"⚡ Latence min/max: {min_latency:.1f}ms / {max_latency:.1f}ms")
        logger.info(f"🎯 Qualité moyenne: {avg_quality:.1f}/10")
        
        # Évaluation objectif latence
        if avg_latency <= self.target_latency:
            logger.info(f"✅ Objectif latence < {self.target_latency}ms: ATTEINT")
            latency_ok = True
        else:
            logger.warning(f"⚠️ Objectif latence < {self.target_latency}ms: DÉPASSÉ (+{avg_latency - self.target_latency:.1f}ms)")
            latency_ok = False
        
        # Évaluation qualité
        quality_ok = avg_quality >= 7.0
        if quality_ok:
            logger.info(f"✅ Objectif qualité ≥ 7.0/10: ATTEINT")
        else:
            logger.warning(f"⚠️ Objectif qualité ≥ 7.0/10: NON ATTEINT")
        
        # Projection pipeline complet
        stt_latency = 833  # ms (validé précédemment)
        tts_latency = 975  # ms (validé précédemment)
        total_pipeline = stt_latency + avg_latency + tts_latency
        
        logger.info("\n📊 PROJECTION PIPELINE COMPLET:")
        logger.info(f"🎤 STT: {stt_latency}ms")
        logger.info(f"🤖 LLM: {avg_latency:.1f}ms")
        logger.info(f"🔊 TTS: {tts_latency}ms")
        logger.info(f"📊 TOTAL: {total_pipeline:.1f}ms ({total_pipeline/1000:.2f}s)")
        
        pipeline_ok = total_pipeline <= 2500  # 2.5s objectif réaliste
        if pipeline_ok:
            logger.info("✅ Objectif pipeline < 2.5s: ATTEINT")
        else:
            logger.warning(f"⚠️ Objectif pipeline < 2.5s: DÉPASSÉ (+{(total_pipeline-2500)/1000:.2f}s)")
        
        # Verdict final
        overall_success = success_rate >= 90 and quality_ok and pipeline_ok
        
        if overall_success:
            logger.info("\n🎊 VALIDATION NOUS-HERMES-2-MISTRAL-7B-DPO: RÉUSSIE")
            logger.info("🚀 Modèle APPROUVÉ pour SuperWhisper V6")
        else:
            logger.warning("\n⚠️ VALIDATION NOUS-HERMES-2-MISTRAL-7B-DPO: CONDITIONNELLE")
            logger.info("🔧 Optimisations recommandées")
        
        # Sauvegarde rapport
        self.save_validation_report({
            "model_name": self.model_name,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "avg_quality": avg_quality,
            "total_pipeline": total_pipeline,
            "approved": overall_success,
            "results": successful_results
        })
        
        return overall_success
    
    def save_validation_report(self, report_data: Dict):
        """Sauvegarder le rapport de validation"""
        import json
        
        report_path = "docs/VALIDATION_NOUS_HERMES_REPORT.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 🦙 RAPPORT VALIDATION NOUS-HERMES-2-MISTRAL-7B-DPO\n\n")
                f.write("## 📊 RÉSULTATS VALIDATION\n\n")
                f.write(f"- **Modèle**: {report_data['model_name']}\n")
                f.write(f"- **Taux de succès**: {report_data['success_rate']:.1f}%\n")
                f.write(f"- **Latence moyenne**: {report_data['avg_latency']:.1f}ms\n")
                f.write(f"- **Qualité moyenne**: {report_data['avg_quality']:.1f}/10\n")
                f.write(f"- **Pipeline total**: {report_data['total_pipeline']:.1f}ms\n")
                f.write(f"- **Statut**: {'✅ APPROUVÉ' if report_data['approved'] else '⚠️ CONDITIONNEL'}\n\n")
                
                f.write("## 🔍 DÉTAILS TESTS\n\n")
                for result in report_data['results'][:5]:  # Top 5 résultats
                    f.write(f"### {result['category'].title()} ({result['config']})\n")
                    f.write(f"- **Latence**: {result['latency']:.1f}ms\n")
                    f.write(f"- **Qualité**: {result['quality_score']:.1f}/10\n")
                    f.write(f"- **Réponse**: \"{result['response'][:100]}...\"\n\n")
            
            logger.info(f"📄 Rapport sauvegardé: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde rapport: {e}")

async def main():
    """Fonction principale de validation"""
    try:
        validator = NousHermesValidator()
        success = await validator.run_comprehensive_validation()
        
        if success:
            print("\n🎊 VALIDATION TERMINÉE AVEC SUCCÈS")
            print("🚀 Nous-Hermes-2-Mistral-7B-DPO approuvé pour SuperWhisper V6")
            return 0
        else:
            print("\n⚠️ VALIDATION TERMINÉE AVEC RÉSERVES")
            print("🔧 Optimisations recommandées avant production")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Erreur validation: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 