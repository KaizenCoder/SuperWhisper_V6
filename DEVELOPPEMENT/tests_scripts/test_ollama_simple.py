#!/usr/bin/env python3
"""
🔧 TEST SIMPLE OLLAMA API
Test rapide pour identifier le problème HTTP 404
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import requests
import json

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
print("🔧 Test Simple API Ollama")
print("=" * 50)

def test_ollama_endpoint(url, data=None, method="GET"):
    """Test un endpoint Ollama"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=30)
        
        print(f"🔗 {method} {url}")
        print(f"  📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  ✅ Succès")
            if response.text:
                try:
                    result = response.json()
                    if isinstance(result, dict) and 'response' in result:
                        print(f"  💬 Réponse: {result['response'][:100]}...")
                    elif isinstance(result, list) and len(result) > 0:
                        print(f"  📦 Éléments: {len(result)}")
                    else:
                        print(f"  📄 Données: {str(result)[:100]}...")
                except:
                    print(f"  📄 Texte: {response.text[:100]}...")
        else:
            print(f"  ❌ Erreur: {response.status_code}")
            print(f"  📄 Texte: {response.text[:200]}")
        
        return response.status_code == 200, response
        
    except Exception as e:
        print(f"  💥 Exception: {e}")
        return False, None

# Tests des endpoints
print("\n1️⃣ Test Santé Ollama")
test_ollama_endpoint("http://127.0.0.1:11434/api/tags")

print("\n2️⃣ Test API Generate (format natif)")
data_native = {
    "model": "nous-hermes-2-mistral-7b-dpo:latest",
    "prompt": "Quelle est la capitale de la France?",
    "stream": False
}
success_native, resp_native = test_ollama_endpoint("http://127.0.0.1:11434/api/generate", data_native, "POST")

print("\n3️⃣ Test API Chat (format OpenAI-like)")
data_openai = {
    "model": "nous-hermes-2-mistral-7b-dpo:latest",
    "messages": [{"role": "user", "content": "Quelle est la capitale de la France?"}],
    "stream": False
}
success_openai, resp_openai = test_ollama_endpoint("http://127.0.0.1:11434/v1/chat/completions", data_openai, "POST")

print("\n4️⃣ Test API Chat (endpoint alternatif)")
success_chat, resp_chat = test_ollama_endpoint("http://127.0.0.1:11434/api/chat", data_openai, "POST")

# Résumé
print("\n" + "=" * 50)
print("📊 RÉSUMÉ DES TESTS")
print("=" * 50)
print(f"✅ API Native (/api/generate): {'OUI' if success_native else 'NON'}")
print(f"✅ API OpenAI (/v1/chat/completions): {'OUI' if success_openai else 'NON'}")
print(f"✅ API Chat (/api/chat): {'OUI' if success_chat else 'NON'}")

if success_native:
    print("\n🎯 SOLUTION: Utiliser l'API native /api/generate")
elif success_openai:
    print("\n🎯 SOLUTION: Utiliser l'API OpenAI /v1/chat/completions")
elif success_chat:
    print("\n🎯 SOLUTION: Utiliser l'API chat /api/chat")
else:
    print("\n❌ PROBLÈME: Aucune API ne fonctionne")

print("\n🔧 Test terminé!") 