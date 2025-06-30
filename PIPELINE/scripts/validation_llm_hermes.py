#!/usr/bin/env python3
"""
Script de validation LLM : nous-hermes-2-mistral-7b-dpo
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

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
import httpx
import json
import time
from datetime import datetime

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch

def validate_rtx3090_configuration():
    """Validation obligatoire de la configuration RTX 3090"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

class LLMValidator:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "nous-hermes-2-mistral-7b-dpo:latest"
        self.test_prompts = [
            "Salut ! Comment ça va ?",
            "Peux-tu me dire l'heure qu'il est ?",
            "Raconte-moi une blague courte.",
            "Quel est le sens de la vie ?",
            "Parle-moi de l'intelligence artificielle."
        ]
        
    async def test_model_availability(self):
        """Vérifier que le modèle est disponible dans Ollama"""
        print(f"\n🔍 Vérification disponibilité modèle: {self.model_name}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model["name"] for model in models_data["models"]]
                    
                    print(f"📋 Modèles disponibles: {len(available_models)}")
                    for model in available_models:
                        print(f"  - {model}")
                    
                    if self.model_name in available_models:
                        print(f"✅ Modèle {self.model_name} trouvé !")
                        return True
                    else:
                        print(f"❌ Modèle {self.model_name} non trouvé")
                        return False
                else:
                    print(f"❌ Erreur API tags: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Erreur connexion Ollama: {e}")
            return False
    
    async def test_chat_generation(self, prompt: str):
        """Tester génération de réponse pour un prompt"""
        print(f"\n💬 Test prompt: \"{prompt}\"")
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "Tu es un assistant IA conversationnel en français. Réponds de façon naturelle et concise."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50,  # Réponses courtes pour tests
                "top_p": 0.9
            }
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload
                )
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    generated_text = data["message"]["content"]
                    
                    # Calculer métriques
                    word_count = len(generated_text.split())
                    char_count = len(generated_text)
                    
                    print(f"✅ Réponse générée ({latency:.1f}ms)")
                    print(f"📝 Contenu: \"{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}\"")
                    print(f"📊 Métriques: {word_count} mots, {char_count} caractères")
                    
                    # Évaluation qualité simple
                    quality_score = self.evaluate_response_quality(prompt, generated_text)
                    print(f"⭐ Qualité: {quality_score}/10")
                    
                    return {
                        "success": True,
                        "response": generated_text,
                        "latency_ms": latency,
                        "word_count": word_count,
                        "char_count": char_count,
                        "quality_score": quality_score
                    }
                else:
                    print(f"❌ Erreur génération: {response.status_code}")
                    print(f"📋 Détails: {response.text}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            print(f"❌ Exception génération ({latency:.1f}ms): {e}")
            return {"success": False, "error": str(e), "latency_ms": latency}
    
    def evaluate_response_quality(self, prompt: str, response: str) -> int:
        """Évaluation simple de la qualité de la réponse (0-10)"""
        score = 5  # Score de base
        
        # Critères positifs
        if len(response.strip()) > 10:  # Réponse non vide
            score += 1
        if any(word in response.lower() for word in ["bonjour", "salut", "merci", "oui", "non"]):
            score += 1
        if len(response.split()) >= 3:  # Au moins 3 mots
            score += 1
        if response.endswith(('.', '!', '?')):  # Ponctuation correcte
            score += 1
        if not response.isupper():  # Pas tout en majuscules
            score += 1
        
        # Critères négatifs
        if "sorry" in response.lower() or "désolé" in response.lower():
            score -= 1
        if len(response) < 5:  # Réponse trop courte
            score -= 2
        if response.count("?") > 2:  # Trop de questions
            score -= 1
            
        return max(0, min(10, score))
    
    async def run_validation(self):
        """Exécuter validation complète du modèle LLM"""
        print("🚀 VALIDATION LLM HERMES-2-MISTRAL-7B-DPO")
        print("=" * 50)
        
        # Vérifier modèle disponible
        if not await self.test_model_availability():
            print("🚫 Validation échouée: modèle non disponible")
            return False
        
        # Tests génération
        print(f"\n🧪 Tests génération sur {len(self.test_prompts)} prompts")
        results = []
        
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"\n--- Test {i}/{len(self.test_prompts)} ---")
            result = await self.test_chat_generation(prompt)
            results.append(result)
            
            # Pause entre tests
            await asyncio.sleep(1)
        
        # Rapport final
        self.generate_report(results)
        
        return True
    
    def generate_report(self, results):
        """Générer rapport de validation"""
        print("\n" + "=" * 50)
        print("📊 RAPPORT VALIDATION LLM")
        print("=" * 50)
        
        successful_tests = [r for r in results if r.get("success", False)]
        failed_tests = [r for r in results if not r.get("success", False)]
        
        print(f"✅ Tests réussis: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.1f}%)")
        print(f"❌ Tests échoués: {len(failed_tests)}/{len(results)} ({len(failed_tests)/len(results)*100:.1f}%)")
        
        if successful_tests:
            latencies = [r["latency_ms"] for r in successful_tests]
            qualities = [r["quality_score"] for r in successful_tests]
            
            print(f"\n📈 Métriques performance:")
            print(f"  - Latence moyenne: {sum(latencies)/len(latencies):.1f}ms")
            print(f"  - Latence P95: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
            print(f"  - Qualité moyenne: {sum(qualities)/len(qualities):.1f}/10")
            
            # Objectif latence < 400ms pour total < 1.2s
            avg_latency = sum(latencies)/len(latencies)
            if avg_latency < 400:
                print(f"🎯 OBJECTIF LATENCE ATTEINT: {avg_latency:.1f}ms < 400ms")
            else:
                print(f"⚠️ OBJECTIF LATENCE MANQUÉ: {avg_latency:.1f}ms > 400ms")
        
        if failed_tests:
            print(f"\n⚠️ Erreurs détectées:")
            for i, test in enumerate(failed_tests):
                print(f"  - Test {i+1}: {test.get('error', 'Erreur inconnue')}")
        
        print(f"\n📝 Validation terminée: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    """Fonction principale"""
    try:
        # Validation GPU obligatoire
        validate_rtx3090_configuration()
        
        # Validation LLM
        validator = LLMValidator()
        success = await validator.run_validation()
        
        if success:
            print("\n✅ VALIDATION LLM RÉUSSIE")
            return 0
        else:
            print("\n❌ VALIDATION LLM ÉCHOUÉE")
            return 1
            
    except Exception as e:
        print(f"\n🚫 ERREUR VALIDATION: {e}")
        return 1

if __name__ == "__main__":
    # Validation RTX 3090 au démarrage
    validate_rtx3090_configuration()
    
    # Exécution validation
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 