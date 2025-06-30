#!/usr/bin/env python3
"""
Test direct API Ollama pour diagnostiquer le problème
"""

import requests
import json

def test_ollama_api():
    """Test direct de l'API Ollama"""
    
    print("🔍 Test API Ollama Direct")
    print("=" * 40)
    
    try:
        # 1. Vérifier liste modèles
        print("1. Liste des modèles:")
        response = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for model in models:
                print(f"   📦 {model['name']}")
            print(f"   Total: {len(models)} modèles")
        else:
            print(f"   ❌ Erreur: {response.status_code}")
            return
        
        # 2. Tester nous-hermes
        print("\n2. Test nous-hermes:")
        model_name = "nous-hermes-2-mistral-7b-dpo:latest"
        
        test_data = {
            "model": model_name,
            "prompt": "Quelle est la capitale de la France ? Répondez en une phrase.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50
            }
        }
        
        print(f"   Modèle: {model_name}")
        print(f"   URL: http://127.0.0.1:11434/api/generate")
        
        response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json=test_data,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', 'Pas de réponse')
            print(f"   ✅ Réponse: {response_text}")
        else:
            print(f"   ❌ Erreur: {response.text}")
        
        # 3. Test avec nom court
        print("\n3. Test avec nom court 'nous-hermes':")
        test_data_short = {
            "model": "nous-hermes",
            "prompt": "Capitale de France ?",
            "stream": False
        }
        
        response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json=test_data_short,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Réponse: {result.get('response', 'Pas de réponse')}")
        else:
            print(f"   ❌ Erreur: {response.text}")
            
    except Exception as e:
        print(f"❌ Erreur test: {e}")

if __name__ == "__main__":
    test_ollama_api()