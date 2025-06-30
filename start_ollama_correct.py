#!/usr/bin/env python3
"""
Démarrage correct Ollama avec modèles D:/modeles_llm
"""

import subprocess
import time
import os
import requests

def start_ollama_service():
    """Démarre le service Ollama correctement"""
    print("🚀 Configuration et démarrage Ollama...")
    
    try:
        # Arrêter tous les processus Ollama existants
        print("🛑 Arrêt processus Ollama existants...")
        subprocess.run(['cmd.exe', '/c', 'taskkill /F /IM ollama.exe /T'], 
                      capture_output=True)
        subprocess.run(['cmd.exe', '/c', 'taskkill /F /IM "ollama app.exe" /T'], 
                      capture_output=True)
        time.sleep(3)
        
        # Définir les variables d'environnement
        env_vars = {
            'OLLAMA_MODELS': 'D:\\modeles_llm',
            'OLLAMA_HOST': '127.0.0.1:11434',
            'OLLAMA_ORIGINS': '*'
        }
        
        print(f"📁 OLLAMA_MODELS: {env_vars['OLLAMA_MODELS']}")
        
        # Créer commande de démarrage
        cmd = [
            'cmd.exe', '/c',
            f'set OLLAMA_MODELS={env_vars["OLLAMA_MODELS"]} && '
            f'set OLLAMA_HOST={env_vars["OLLAMA_HOST"]} && '
            f'set OLLAMA_ORIGINS={env_vars["OLLAMA_ORIGINS"]} && '
            'C:\\Users\\Utilisateur\\AppData\\Local\\Programs\\Ollama\\ollama.exe serve'
        ]
        
        print("⏳ Démarrage du service Ollama...")
        
        # Démarrer Ollama en arrière-plan
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Attendre démarrage
        print("⏱️ Attente initialisation (15s)...")
        for i in range(15):
            time.sleep(1)
            print(f"  {i+1}/15s", end='\r')
        print()
        
        return process
        
    except Exception as e:
        print(f"❌ Erreur démarrage: {e}")
        return None

def test_ollama_connection():
    """Test la connexion Ollama"""
    print("🔍 Test connexion Ollama...")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get('http://127.0.0.1:11434/api/tags', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✅ Ollama connecté - {len(models)} modèles disponibles")
                
                for model in models[:3]:  # Afficher les 3 premiers
                    name = model.get('name', 'unknown')
                    size_mb = model.get('size', 0) / (1024*1024)
                    print(f"  📦 {name} ({size_mb:.0f}MB)")
                
                return True, models
            else:
                print(f"⚠️ Status {response.status_code} - Tentative {attempt+1}/{max_retries}")
                
        except requests.exceptions.ConnectionError:
            print(f"⏳ Connexion impossible - Tentative {attempt+1}/{max_retries}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(3)
    
    print("❌ Connexion Ollama échouée")
    return False, []

def test_model_simple(models):
    """Test simple d'un modèle"""
    if not models:
        print("⚠️ Aucun modèle à tester")
        return False
    
    model_name = models[0]['name']
    print(f"🧠 Test modèle: {model_name}")
    
    try:
        data = {
            "model": model_name,
            "prompt": "Bonjour, répondez brièvement en français.",
            "stream": False,
            "options": {
                "num_predict": 50
            }
        }
        
        response = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '').strip()
            print(f"✅ Réponse modèle: '{response_text[:100]}...'")
            return True
        else:
            print(f"❌ Erreur modèle: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test modèle: {e}")
        return False

def main():
    print("🔧 SuperWhisper V6 - Configuration Ollama")
    print("=" * 60)
    
    # Démarrer service
    process = start_ollama_service()
    if not process:
        print("❌ Échec démarrage Ollama")
        return
    
    # Tester connexion
    connected, models = test_ollama_connection()
    
    if connected:
        # Tester un modèle
        model_works = test_model_simple(models)
        
        print("\n📊 RÉSULTATS:")
        print(f"  🚀 Ollama démarré: ✅")
        print(f"  🔗 Connexion: ✅")
        print(f"  📦 Modèles: {len(models)}")
        print(f"  🧠 Test modèle: {'✅' if model_works else '❌'}")
        
        if model_works:
            print("\n🎉 OLLAMA OPÉRATIONNEL POUR SUPERWHISPER V6!")
            print("   Le LLM peut maintenant être utilisé dans le pipeline.")
        else:
            print("\n⚠️ Ollama connecté mais modèle non fonctionnel")
    else:
        print("\n❌ ÉCHEC CONFIGURATION OLLAMA")
        print("   Vérifiez l'installation et les modèles dans D:/modeles_llm")
    
    # Laisser Ollama tourner
    print(f"\n🔄 Ollama reste actif (PID: {process.pid})")
    print("   Utiliser Ctrl+C dans le terminal Ollama pour arrêter")

if __name__ == "__main__":
    main()