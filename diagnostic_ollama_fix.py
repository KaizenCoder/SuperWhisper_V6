#!/usr/bin/env python3
"""
🔧 DIAGNOSTIC OLLAMA - Résolution HTTP 404
Diagnostic complet et correction de l'API Ollama pour SuperWhisper V6
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import time
import json
import pathlib
import asyncio
import subprocess
from typing import Dict, Any, Optional

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    current_file = pathlib.Path(__file__).resolve()
    
    # Chercher le répertoire racine
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.chdir(project_root)
    print(f"📁 Project Root: {project_root}")
    return project_root

_PROJECT_ROOT = _setup_portable_environment()

# Maintenant imports normaux...
import requests
import httpx

class OllamaDiagnostic:
    """Diagnostic complet de l'API Ollama"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_target = "nous-hermes-2-mistral-7b-dpo:latest"
        self.available_models = []
        
    def start_ollama_service(self):
        """Démarre le service Ollama"""
        print("\n🚀 ÉTAPE 1: Démarrage Ollama")
        print("=" * 50)
        
        try:
            # Tuer processus existant
            subprocess.run(['cmd.exe', '/c', 'taskkill /F /IM ollama.exe'], 
                          capture_output=True, shell=True)
            time.sleep(2)
            
            # Configurer environnement
            env = os.environ.copy()
            env['OLLAMA_MODELS'] = 'D:\\modeles_llm'
            
            print("📁 OLLAMA_MODELS: D:\\modeles_llm")
            
            # Démarrer Ollama en arrière-plan
            process = subprocess.Popen(
                ['ollama', 'serve'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            
            print("⏳ Attente démarrage Ollama (10s)...")
            time.sleep(10)
            
            return process
            
        except Exception as e:
            print(f"❌ Erreur démarrage Ollama: {e}")
            return None
    
    def check_ollama_health(self):
        """Vérification santé Ollama"""
        print("\n🔍 ÉTAPE 2: Vérification Santé Ollama")
        print("=" * 50)
        
        endpoints_to_test = [
            "/api/tags",
            "/api/version", 
            "/v1/models",
            "/health"
        ]
        
        results = {}
        
        for endpoint in endpoints_to_test:
            url = f"{self.base_url}{endpoint}"
            try:
                print(f"🔗 Test: {url}")
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    print(f"  ✅ Status: {response.status_code}")
                    try:
                        data = response.json()
                        if endpoint == "/api/tags" and "models" in data:
                            self.available_models = [m["name"] for m in data["models"]]
                            print(f"  📦 Modèles trouvés: {len(self.available_models)}")
                        results[endpoint] = {"status": "OK", "data": data}
                    except:
                        results[endpoint] = {"status": "OK", "data": response.text}
                else:
                    print(f"  ❌ Status: {response.status_code}")
                    results[endpoint] = {"status": "ERROR", "code": response.status_code}
                    
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
                results[endpoint] = {"status": "FAILED", "error": str(e)}
        
        return results
    
    def test_api_formats(self):
        """Test différents formats d'API"""
        print("\n🧪 ÉTAPE 3: Test Formats API")
        print("=" * 50)
        
        if not self.available_models:
            print("❌ Aucun modèle disponible - impossible de tester")
            return {}
        
        # Choisir le modèle à tester
        target_model = None
        for model in self.available_models:
            if "nous-hermes" in model.lower() or "mistral" in model.lower():
                target_model = model
                break
        
        if not target_model:
            target_model = self.available_models[0]
        
        print(f"🎯 Modèle de test: {target_model}")
        
        test_prompt = "Quelle est la capitale de la France ?"
        
        # Format 1: API native Ollama /api/generate
        print("\n📝 Format 1: API Native Ollama (/api/generate)")
        result1 = self._test_native_api(target_model, test_prompt)
        
        # Format 2: API OpenAI compatible /v1/chat/completions
        print("\n📝 Format 2: API OpenAI Compatible (/v1/chat/completions)")
        result2 = self._test_openai_api(target_model, test_prompt)
        
        # Format 3: API chat Ollama /api/chat
        print("\n📝 Format 3: API Chat Ollama (/api/chat)")
        result3 = self._test_chat_api(target_model, test_prompt)
        
        return {
            "native_api": result1,
            "openai_api": result2, 
            "chat_api": result3,
            "model_used": target_model
        }
    
    def _test_native_api(self, model: str, prompt: str) -> Dict[str, Any]:
        """Test API native Ollama"""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 50
                }
            }
            
            url = f"{self.base_url}/api/generate"
            print(f"🔗 URL: {url}")
            print(f"📤 Données: {json.dumps(data, indent=2)}")
            
            response = requests.post(url, json=data, timeout=30)
            
            print(f"📥 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                print(f"✅ Réponse: {response_text[:100]}...")
                return {"success": True, "response": response_text, "status": 200}
            else:
                print(f"❌ Erreur: {response.text}")
                return {"success": False, "error": response.text, "status": response.status_code}
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_openai_api(self, model: str, prompt: str) -> Dict[str, Any]:
        """Test API OpenAI compatible"""
        try:
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 50,
                "temperature": 0.7,
                "stream": False
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            print(f"🔗 URL: {url}")
            print(f"📤 Données: {json.dumps(data, indent=2)}")
            
            response = requests.post(url, json=data, timeout=30)
            
            print(f"📥 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content']
                print(f"✅ Réponse: {response_text[:100]}...")
                return {"success": True, "response": response_text, "status": 200}
            else:
                print(f"❌ Erreur: {response.text}")
                return {"success": False, "error": response.text, "status": response.status_code}
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_chat_api(self, model: str, prompt: str) -> Dict[str, Any]:
        """Test API chat Ollama"""
        try:
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 50
                }
            }
            
            url = f"{self.base_url}/api/chat"
            print(f"🔗 URL: {url}")
            print(f"📤 Données: {json.dumps(data, indent=2)}")
            
            response = requests.post(url, json=data, timeout=30)
            
            print(f"📥 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['message']['content']
                print(f"✅ Réponse: {response_text[:100]}...")
                return {"success": True, "response": response_text, "status": 200}
            else:
                print(f"❌ Erreur: {response.text}")
                return {"success": False, "error": response.text, "status": response.status_code}
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_fix_recommendations(self, test_results: Dict[str, Any]):
        """Génère les recommandations de correction"""
        print("\n🔧 ÉTAPE 4: Recommandations de Correction")
        print("=" * 50)
        
        working_apis = []
        
        # Analyser les résultats
        for api_name, result in test_results.items():
            if api_name == "model_used":
                continue
                
            if result.get("success", False):
                working_apis.append(api_name)
                print(f"✅ {api_name}: FONCTIONNE")
            else:
                print(f"❌ {api_name}: ÉCHOUE - {result.get('error', 'Erreur inconnue')}")
        
        print(f"\n🎯 APIs fonctionnelles: {working_apis}")
        
        if "native_api" in working_apis:
            print("\n💡 RECOMMANDATION: Utiliser l'API Native Ollama")
            self._show_native_fix()
        elif "chat_api" in working_apis:
            print("\n💡 RECOMMANDATION: Utiliser l'API Chat Ollama")
            self._show_chat_fix()
        elif "openai_api" in working_apis:
            print("\n💡 RECOMMANDATION: Utiliser l'API OpenAI Compatible")
            self._show_openai_fix()
        else:
            print("\n❌ AUCUNE API FONCTIONNELLE - Problème Ollama")
            self._show_troubleshooting()
    
    def _show_native_fix(self):
        """Montre la correction pour API native"""
        print("""
🔧 CORRECTION LLM MANAGER - API Native:

Dans LLM/llm_manager_enhanced.py, méthode _generate_ollama():

async def _generate_ollama(self, user_input: str, max_tokens: int, temperature: float) -> str:
    try:
        import httpx
        
        # ✅ FORMAT CORRECT API NATIVE OLLAMA
        data = {
            "model": getattr(self, 'actual_model_name', self.config.get('model')),
            "prompt": f"{self.system_prompt}\\n\\nUser: {user_input}\\nAssistant:",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                'http://127.0.0.1:11434/api/generate',  # ✅ ENDPOINT CORRECT
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        self.logger.error(f"Erreur Ollama: {e}")
        return None
""")
    
    def _show_chat_fix(self):
        """Montre la correction pour API chat"""
        print("""
🔧 CORRECTION LLM MANAGER - API Chat:

Dans LLM/llm_manager_enhanced.py, méthode _generate_ollama():

async def _generate_ollama(self, user_input: str, max_tokens: int, temperature: float) -> str:
    try:
        import httpx
        
        # ✅ FORMAT CORRECT API CHAT OLLAMA
        data = {
            "model": getattr(self, 'actual_model_name', self.config.get('model')),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                'http://127.0.0.1:11434/api/chat',  # ✅ ENDPOINT CORRECT
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['message']['content'].strip()
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        self.logger.error(f"Erreur Ollama: {e}")
        return None
""")
    
    def _show_troubleshooting(self):
        """Montre le troubleshooting"""
        print("""
🚨 TROUBLESHOOTING OLLAMA:

1. Vérifier installation:
   ollama --version

2. Redémarrer service:
   ollama serve

3. Vérifier modèles:
   ollama list

4. Télécharger modèle si absent:
   ollama pull nous-hermes-2-mistral-7b-dpo

5. Tester manuellement:
   ollama run nous-hermes-2-mistral-7b-dpo "Bonjour"

6. Vérifier port:
   netstat -an | findstr 11434
""")

def main():
    """Fonction principale de diagnostic"""
    print("🔧 DIAGNOSTIC OLLAMA - SuperWhisper V6")
    print("=" * 60)
    
    diagnostic = OllamaDiagnostic()
    
    # Étape 1: Démarrer Ollama
    process = diagnostic.start_ollama_service()
    if not process:
        print("❌ Impossible de démarrer Ollama")
        return
    
    try:
        # Étape 2: Vérifier santé
        health_results = diagnostic.check_ollama_health()
        
        # Étape 3: Tester APIs
        api_results = diagnostic.test_api_formats()
        
        # Étape 4: Recommandations
        diagnostic.generate_fix_recommendations(api_results)
        
        # Sauvegarde rapport
        report = {
            "timestamp": time.time(),
            "health_check": health_results,
            "api_tests": api_results,
            "models_available": diagnostic.available_models
        }
        
        with open("diagnostic_ollama_rapport.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Rapport sauvegardé: diagnostic_ollama_rapport.json")
        
    finally:
        # Nettoyer processus
        if process:
            try:
                process.terminate()
            except:
                pass

if __name__ == "__main__":
    main() 