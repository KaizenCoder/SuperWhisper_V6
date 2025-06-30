#!/usr/bin/env python3
"""
Test direct Ollama - Démarrage et test des modèles
"""

import subprocess
import time
import requests
import json
import os

def start_ollama():
    """Démarre Ollama avec le bon répertoire de modèles"""
    print("🚀 Démarrage Ollama...")
    
    env = os.environ.copy()
    env['OLLAMA_MODELS'] = 'D:\\modeles_llm'
    
    try:
        # Tuer processus existant
        subprocess.run(['cmd.exe', '/c', 'taskkill /F /IM ollama.exe'], 
                      capture_output=True)
        time.sleep(2)
        
        # Démarrer Ollama
        process = subprocess.Popen(
            ['cmd.exe', '/c', 'ollama serve'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("⏳ Attente démarrage Ollama...")
        time.sleep(10)
        
        return process
        
    except Exception as e:
        print(f"❌ Erreur démarrage: {e}")
        return None

def test_ollama():
    """Test la connexion et les modèles Ollama"""
    print("🔍 Test connexion Ollama...")
    
    try:
        # Test santé
        response = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama accessible - {len(models)} modèles")
            
            for model in models:
                print(f"  📦 {model['name']} ({model['size']} bytes)")
            
            return models
        else:
            print(f"❌ Status: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return []

def test_model_inference(model_name):
    """Test d'inférence avec un modèle"""
    print(f"🧠 Test inférence: {model_name}")
    
    try:
        data = {
            "model": model_name,
            "prompt": "Bonjour, comment allez-vous ?",
            "stream": False
        }
        
        response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Réponse: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Erreur inférence: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    print("🔧 Test Ollama SuperWhisper V6")
    print("=" * 50)
    
    # Démarrer Ollama
    process = start_ollama()
    
    # Tester connexion
    models = test_ollama()
    
    if models:
        # Tester premier modèle disponible
        first_model = models[0]['name']
        test_model_inference(first_model)
    else:
        print("❌ Aucun modèle trouvé")
    
    print("\n📋 Résumé:")
    print(f"  - Ollama démarré: {'✅' if process else '❌'}")
    print(f"  - Modèles disponibles: {len(models)}")
    
    if process:
        input("\n⏸️ Appuyez sur Entrée pour arrêter Ollama...")
        process.terminate()

if __name__ == "__main__":
    main()