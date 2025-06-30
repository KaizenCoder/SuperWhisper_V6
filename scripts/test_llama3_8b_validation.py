#!/usr/bin/env python3
"""
🦙 VALIDATION LLAMA3 8B INSTRUCT Q6_K - SUPERWHISPER V6
=====================================================
Script validation spécialisé pour Llama3 8B Instruct Q6_K local

MODÈLE CIBLE:
- Nom: llama3:8b-instruct-q6_k
- Fichier: Meta-Llama-3-8B-Instruct-Q6_K.gguf
- Localisation: D:\modeles_llm\lmstudio-community\Meta-Llama-3-8B-Instruct-GGUF\
- Quantization: Q6_K (haute qualité)

MISSION:
- Validation latence < 400ms pour pipeline < 2.5s
- Tests conversation française instruction-tuned
- Configuration RTX 3090 optimisée
- Sélection paramètres optimaux

Usage: python scripts/test_llama3_8b_validation.py

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
from typing import Dict, Any, List
import httpx

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Configuration logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("llama3_validation")

class Llama3ValidationTester:
    """Validateur spécialisé Llama3 8B Instruct Q6_K"""
    
    def __init__(self):
        self.model_name = "llama3:8b-instruct-q6_k"
        self.model_path = "D:\\modeles_llm\\lmstudio-community\\Meta-Llama-3-8B-Instruct-GGUF\\Meta-Llama-3-8B-Instruct-Q6_K.gguf"
        
        # Endpoints possibles pour Llama3 8B
        self.endpoints = [
            {"url": "http://localhost:1234/v1", "name": "LM Studio", "api_type": "openai"},
            {"url": "http://localhost:11434", "name": "Ollama", "api_type": "ollama"},
            {"url": "http://localhost:8080", "name": "llama.cpp", "api_type": "llamacpp"}
        ]
        
        self.timeout = 30
        
        # Tests conversation française optimisés pour instruction-tuned
        self.conversation_tests = [
            {
                "prompt": "Bonjour ! Comment allez-vous ?",
                "expected_type": "greeting",
                "max_tokens": 20
            },
            {
                "prompt": "Quelle heure est-il ?",
                "expected_type": "time_request", 
                "max_tokens": 15
            },
            {
                "prompt": "Merci beaucoup !",
                "expected_type": "gratitude",
                "max_tokens": 10
            },
            {
                "prompt": "Pouvez-vous m'aider ?",
                "expected_type": "help_request",
                "max_tokens": 25
            },
            {
                "prompt": "Au revoir !",
                "expected_type": "farewell",
                "max_tokens": 10
            }
        ]
    
    async def check_model_availability(self) -> Dict[str, Any]:
        """Vérification disponibilité modèle Llama3 8B"""
        logger.info("🔍 Vérification disponibilité Llama3 8B Instruct Q6_K...")
        
        # Vérification fichier local
        if os.path.exists(self.model_path):
            file_size_gb = os.path.getsize(self.model_path) / (1024**3)
            logger.info(f"✅ Fichier modèle trouvé: {file_size_gb:.1f}GB")
        else:
            logger.warning(f"⚠️ Fichier modèle non trouvé: {self.model_path}")
        
        # Test endpoints
        available_endpoint = None
        for endpoint in self.endpoints:
            try:
                logger.info(f"🔍 Test endpoint {endpoint['name']}: {endpoint['url']}")
                
                async with httpx.AsyncClient(timeout=10) as client:
                    if endpoint['api_type'] == 'openai':
                        # Test OpenAI compatible (LM Studio)
                        response = await client.get(f"{endpoint['url']}/models")
                    elif endpoint['api_type'] == 'ollama':
                        # Test Ollama
                        response = await client.get(f"{endpoint['url']}/api/tags")
                    else:
                        # Test llama.cpp
                        response = await client.get(f"{endpoint['url']}/health")
                    
                    if response.status_code == 200:
                        logger.info(f"✅ {endpoint['name']} disponible")
                        available_endpoint = endpoint
                        break
                        
            except Exception as e:
                logger.info(f"❌ {endpoint['name']} non disponible: {e}")
        
        return {
            "model_file_exists": os.path.exists(self.model_path),
            "available_endpoint": available_endpoint
        }
    
    async def test_llama3_inference(self, endpoint: Dict, prompt: str, max_tokens: int = 20) -> Dict[str, Any]:
        """Test inférence Llama3 8B avec optimisations"""
        start_time = time.time()
        
        try:
            if endpoint['api_type'] == 'openai':
                # API OpenAI compatible (LM Studio)
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "Tu es un assistant vocal français. Réponds de manière très concise et naturelle."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1,
                    "stop": [".", "!", "?", "\n"]
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{endpoint['url']}/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    response_text = data['choices'][0]['message']['content'].strip()
                    
            elif endpoint['api_type'] == 'ollama':
                # API Ollama
                payload = {
                    "model": self.model_name,
                    "prompt": f"Système: Tu es un assistant vocal français. Réponds de manière très concise et naturelle.\n\nUtilisateur: {prompt}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_tokens,
                        "top_p": 0.8,
                        "top_k": 40,
                        "repeat_penalty": 1.1,
                        "stop": [".", "!", "?", "\n", "Utilisateur:"]
                    }
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{endpoint['url']}/api/generate",
                        json=payload
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    response_text = data.get('response', '').strip()
            
            else:
                # llama.cpp server
                payload = {
                    "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nTu es un assistant vocal français. Réponds de manière très concise et naturelle.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "n_predict": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "stop": ["<|eot_id|>", ".", "!", "?", "\n"]
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{endpoint['url']}/completion",
                        json=payload
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    response_text = data.get('content', '').strip()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'latency_ms': latency_ms,
                'response_text': response_text,
                'tokens': len(response_text.split()),
                'endpoint': endpoint['name']
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                'success': False,
                'latency_ms': latency_ms,
                'error': str(e),
                'endpoint': endpoint['name']
            }
    
    def evaluate_llama3_response(self, prompt: str, response: str, expected_type: str) -> float:
        """Évaluation qualité réponse Llama3 (0-10)"""
        if not response or len(response.strip()) < 2:
            return 0.0
        
        score = 6.0  # Score base pour Llama3 8B
        
        # Critères généraux
        if len(response) >= 5 and len(response) <= 50:  # Longueur appropriée
            score += 1.0
        
        # Détection français
        french_words = ['bonjour', 'merci', 'oui', 'non', 'je', 'vous', 'est', 'avec', 'pour', 'dans', 'sur']
        if any(word in response.lower() for word in french_words):
            score += 1.5
        
        # Évaluation contextuelle selon type attendu
        response_lower = response.lower()
        
        if expected_type == "greeting" and any(word in response_lower for word in ['bonjour', 'salut', 'bonsoir', 'bien', 'ça va']):
            score += 1.5
        elif expected_type == "time_request" and any(word in response_lower for word in ['heure', 'temps', 'maintenant', 'actuellement']):
            score += 1.5
        elif expected_type == "gratitude" and any(word in response_lower for word in ['rien', 'plaisir', 'service', 'bienvenue']):
            score += 1.5
        elif expected_type == "help_request" and any(word in response_lower for word in ['aide', 'aider', 'assistance', 'service', 'oui']):
            score += 1.5
        elif expected_type == "farewell" and any(word in response_lower for word in ['revoir', 'bientôt', 'bonne']):
            score += 1.5
        
        return min(10.0, score)
    
    async def run_llama3_validation(self) -> Dict[str, Any]:
        """Validation complète Llama3 8B Instruct Q6_K"""
        logger.info("🦙 DÉMARRAGE VALIDATION LLAMA3 8B INSTRUCT Q6_K")
        
        # Vérification disponibilité
        availability = await self.check_model_availability()
        
        if not availability['available_endpoint']:
            return {
                'success': False,
                'error': 'Aucun endpoint disponible pour Llama3 8B',
                'model_file_exists': availability['model_file_exists']
            }
        
        endpoint = availability['available_endpoint']
        logger.info(f"🚀 Utilisation endpoint: {endpoint['name']} ({endpoint['url']})")
        
        # Tests conversation
        results = []
        latencies = []
        quality_scores = []
        
        for i, test in enumerate(self.conversation_tests):
            logger.info(f"Test {i+1}/5: '{test['prompt']}'")
            
            result = await self.test_llama3_inference(
                endpoint, 
                test['prompt'], 
                test['max_tokens']
            )
            
            if result['success']:
                quality = self.evaluate_llama3_response(
                    test['prompt'], 
                    result['response_text'], 
                    test['expected_type']
                )
                
                latencies.append(result['latency_ms'])
                quality_scores.append(quality)
                results.append({
                    'prompt': test['prompt'],
                    'response': result['response_text'],
                    'latency_ms': result['latency_ms'],
                    'quality': quality,
                    'tokens': result['tokens']
                })
                
                logger.info(f"   ✅ {result['latency_ms']:.1f}ms - '{result['response_text']}' - Q:{quality:.1f}/10")
            else:
                logger.warning(f"   ❌ Échec: {result['error']}")
                results.append({
                    'prompt': test['prompt'],
                    'error': result['error'],
                    'latency_ms': result['latency_ms']
                })
        
        # Calcul métriques finales
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            avg_quality = sum(quality_scores) / len(quality_scores)
            success_rate = len(latencies) / len(self.conversation_tests)
            
            return {
                'success': True,
                'model_name': self.model_name,
                'endpoint': endpoint['name'],
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'avg_quality': avg_quality,
                'success_rate': success_rate,
                'results': results,
                'model_file_exists': availability['model_file_exists']
            }
        else:
            return {
                'success': False,
                'error': 'Aucun test réussi',
                'results': results
            }

def generate_llama3_report(validation_result: Dict[str, Any]) -> str:
    """Génération rapport validation Llama3 8B"""
    
    report = "="*70 + "\n"
    report += "🦙 RAPPORT VALIDATION LLAMA3 8B INSTRUCT Q6_K - SUPERWHISPER V6\n"
    report += "="*70 + "\n"
    
    if not validation_result['success']:
        report += f"❌ VALIDATION ÉCHOUÉE: {validation_result.get('error', 'Erreur inconnue')}\n"
        report += f"📁 Fichier modèle: {'✅ Trouvé' if validation_result.get('model_file_exists') else '❌ Non trouvé'}\n"
        report += "="*70
        return report
    
    # Métriques principales
    report += f"🏆 MODÈLE: {validation_result['model_name']}\n"
    report += f"🔗 ENDPOINT: {validation_result['endpoint']}\n"
    report += f"⚡ Latence moyenne: {validation_result['avg_latency_ms']:.1f}ms\n"
    report += f"🚀 Latence minimum: {validation_result['min_latency_ms']:.1f}ms\n"
    report += f"⏱️ Latence maximum: {validation_result['max_latency_ms']:.1f}ms\n"
    report += f"🏅 Qualité moyenne: {validation_result['avg_quality']:.1f}/10\n"
    report += f"📈 Taux succès: {validation_result['success_rate']*100:.1f}%\n"
    
    # Évaluation objectifs
    avg_latency = validation_result['avg_latency_ms']
    report += f"\n🎯 ÉVALUATION OBJECTIFS:\n"
    report += f"   Objectif < 400ms: {'✅ ATTEINT' if avg_latency < 400 else '❌ DÉPASSÉ'}\n"
    report += f"   Objectif < 600ms: {'✅ ATTEINT' if avg_latency < 600 else '❌ DÉPASSÉ'}\n"
    
    # Projection pipeline
    stt_latency = 833
    tts_latency = 975
    total_pipeline = stt_latency + avg_latency + tts_latency
    
    report += f"\n📊 PROJECTION PIPELINE COMPLET:\n"
    report += f"   STT: {stt_latency}ms\n"
    report += f"   LLM: {avg_latency:.1f}ms (Llama3 8B)\n"
    report += f"   TTS: {tts_latency}ms\n"
    report += f"   TOTAL: {total_pipeline:.1f}ms\n"
    report += f"   Objectif < 1200ms: {'✅ ATTEINT' if total_pipeline < 1200 else '❌ DÉPASSÉ'}\n"
    report += f"   Objectif < 2500ms: {'✅ ATTEINT' if total_pipeline < 2500 else '❌ DÉPASSÉ'}\n"
    
    # Détail conversations
    report += f"\n💬 EXEMPLES CONVERSATIONS:\n"
    for result in validation_result['results']:
        if 'response' in result:
            report += f"   '{result['prompt']}' → '{result['response']}' ({result['latency_ms']:.0f}ms)\n"
    
    # Recommandations
    report += f"\n🎯 RECOMMANDATIONS:\n"
    if avg_latency < 600 and validation_result['avg_quality'] >= 8.0:
        report += "✅ Llama3 8B validé pour production SuperWhisper V6\n"
        report += "🚀 Qualité et performance excellentes\n"
        report += "🔄 Prêt pour intégration pipeline complet\n"
    elif avg_latency < 800:
        report += "⚠️ Performance acceptable mais optimisation recommandée\n"
        report += "🔧 Réduire max_tokens ou ajuster paramètres\n"
    else:
        report += "❌ Latence trop élevée pour objectifs SuperWhisper V6\n"
        report += "🔧 Optimisation critique requise\n"
    
    report += "="*70
    return report

async def main():
    """Validation principale Llama3 8B Instruct Q6_K"""
    try:
        tester = Llama3ValidationTester()
        validation_result = await tester.run_llama3_validation()
        
        # Génération rapport
        report = generate_llama3_report(validation_result)
        print(report)
        
        # Sauvegarde rapport
        report_path = "docs/VALIDATION_LLAMA3_8B_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# RAPPORT VALIDATION LLAMA3 8B INSTRUCT Q6_K - SUPERWHISPER V6\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"```\n{report}\n```\n")
        
        print(f"\n📁 Rapport sauvegardé: {report_path}")
        
        # Code retour selon performance
        if validation_result['success']:
            avg_latency = validation_result['avg_latency_ms']
            if avg_latency < 600 and validation_result['avg_quality'] >= 8.0:
                print("\n🎊 LLAMA3 8B VALIDÉ - Prêt pour production")
                return 0
            else:
                print("\n⚠️ LLAMA3 8B fonctionnel mais optimisation recommandée")
                return 1
        else:
            print("\n❌ VALIDATION LLAMA3 8B ÉCHOUÉE")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Erreur validation Llama3 8B: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 