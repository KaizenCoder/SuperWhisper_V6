#!/usr/bin/env python3
"""
Script de configuration et test Azure Speech Services pour SuperWhisper V6
🚀 Configuration automatique et test du backend Azure Speech
"""

import os
import sys
import time
import asyncio
import numpy as np
from pathlib import Path

# Configuration portable
def setup_environment():
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt']):
            project_root = parent
            break
    else:
        project_root = current_file.parent.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.chdir(project_root)
    return project_root

PROJECT_ROOT = setup_environment()

def install_dependencies():
    """Installe les dépendances Azure Speech"""
    import subprocess
    
    print("📦 Installation des dépendances Azure Speech...")
    
    dependencies = [
        'azure-cognitiveservices-speech>=1.34.0'
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✅ {dep} installé")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur installation {dep}: {e}")
            return False
    
    return True

def configure_azure_credentials():
    """Configure les credentials Azure"""
    print("\n🔧 Configuration Azure Speech Services")
    print("=" * 50)
    
    # Vérifier variables d'environnement existantes
    speech_key = os.getenv('AZURE_SPEECH_KEY')
    speech_region = os.getenv('AZURE_SPEECH_REGION')
    
    if speech_key and speech_region:
        print(f"✅ Credentials trouvés:")
        print(f"   - Région: {speech_region}")
        print(f"   - Clé: {speech_key[:8]}...")
        return speech_key, speech_region
    
    print("⚠️ Credentials Azure non trouvés dans variables d'environnement")
    print("\n📋 Pour configurer Azure Speech Services:")
    print("1. Créez une ressource Speech dans le portail Azure")
    print("2. Récupérez la clé et la région")
    print("3. Configurez les variables d'environnement:")
    print("   export AZURE_SPEECH_KEY='votre-clé'")
    print("   export AZURE_SPEECH_REGION='francecentral'")
    
    # Configuration interactive
    print("\n🔧 Configuration interactive:")
    speech_key = input("Clé Azure Speech (ou Entrée pour passer): ").strip()
    speech_region = input("Région Azure (défaut: francecentral): ").strip() or "francecentral"
    
    if speech_key:
        # Créer fichier .env local
        env_file = PROJECT_ROOT / '.env'
        with open(env_file, 'a') as f:
            f.write(f"\n# Azure Speech Services\n")
            f.write(f"AZURE_SPEECH_KEY={speech_key}\n")
            f.write(f"AZURE_SPEECH_REGION={speech_region}\n")
        
        print(f"✅ Configuration sauvée dans {env_file}")
        
        # Mettre à jour variables d'environnement pour ce processus
        os.environ['AZURE_SPEECH_KEY'] = speech_key
        os.environ['AZURE_SPEECH_REGION'] = speech_region
        
        return speech_key, speech_region
    
    return None, None

async def test_azure_speech():
    """Test du backend Azure Speech"""
    print("\n🧪 Test Azure Speech Backend")
    print("=" * 30)
    
    try:
        # Import du backend
        from STT.backends.azure_speech_backend import AzureSpeechBackend
        
        # Configuration test
        config = {
            'azure_speech_key': os.getenv('AZURE_SPEECH_KEY'),
            'azure_speech_region': os.getenv('AZURE_SPEECH_REGION', 'francecentral'),
            'language': 'fr-FR',
            'device': 'azure'
        }
        
        # Créer backend
        print("🔧 Création du backend Azure Speech...")
        backend = AzureSpeechBackend(config)
        
        # Health check
        print("🏥 Health check...")
        if backend.health_check():
            print("✅ Backend Azure Speech opérationnel")
        else:
            print("❌ Backend Azure Speech non opérationnel")
            return False
        
        # Test avec audio synthétique
        print("🎵 Test avec audio synthétique...")
        
        # Générer un signal audio test (1 seconde de silence + bruit)
        sample_rate = 16000
        duration = 2.0
        audio_samples = int(duration * sample_rate)
        
        # Audio test : silence + bruit faible
        audio = np.zeros(audio_samples, dtype=np.float32)
        audio[sample_rate:] = np.random.normal(0, 0.01, audio_samples - sample_rate)
        
        # Test transcription
        start_time = time.time()
        result = await backend.transcribe(audio)
        test_time = time.time() - start_time
        
        print(f"📊 Résultat test:")
        print(f"   - Texte: '{result.text}'")
        print(f"   - Confiance: {result.confidence:.2f}")
        print(f"   - Temps: {test_time:.2f}s")
        print(f"   - RTF: {result.rtf:.2f}")
        print(f"   - Succès: {result.success}")
        
        if result.error:
            print(f"   - Erreur: {result.error}")
        
        return result.success
        
    except ImportError as e:
        print(f"❌ Module manquant: {e}")
        print("💡 Installez avec: pip install azure-cognitiveservices-speech")
        return False
    
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False

def create_example_config():
    """Crée un exemple de configuration"""
    config_content = '''
# Configuration SuperWhisper V6 avec Azure Speech Services

import os

# Configuration Azure Speech
AZURE_SPEECH_CONFIG = {
    "backend_type": "azure_speech",
    "enabled": True,
    "azure_speech_key": os.getenv("AZURE_SPEECH_KEY"),
    "azure_speech_region": os.getenv("AZURE_SPEECH_REGION", "francecentral"),
    "language": "fr-FR",
    "continuous_recognition": True,
    "enable_detailed_results": True,
    "enable_word_level_timestamps": True,
    "segmentation_silence_timeout_ms": 500,
    "initial_silence_timeout_ms": 5000
}

# Exemple d'utilisation
async def example_usage():
    from STT.backends.azure_speech_backend import AzureSpeechBackend
    
    # Créer backend
    backend = AzureSpeechBackend(AZURE_SPEECH_CONFIG)
    
    # Pour streaming continu
    await backend.start_continuous_recognition(
        interim_callback=lambda text: print(f"Interim: {text}"),
        final_callback=lambda result: print(f"Final: {result.text}")
    )
    
    # Pour transcription d'un fichier
    # audio = load_audio_file("test.wav")  # Votre fonction de chargement
    # result = await backend.transcribe(audio)
    # print(f"Transcription: {result.text}")
'''
    
    config_file = PROJECT_ROOT / 'config_azure_speech_example.py'
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"📝 Exemple de configuration créé: {config_file}")

def show_performance_comparison():
    """Affiche une comparaison de performance"""
    print("\n📊 Comparaison Performance Azure Speech vs Whisper")
    print("=" * 55)
    print(f"{'Critère':<25} {'Whisper Local':<15} {'Azure Speech':<15}")
    print("-" * 55)
    print(f"{'Latence':<25} {'1-3s':<15} {'<300ms':<15}")
    print(f"{'RTF (Real-Time Factor)':<25} {'0.3-1.0':<15} {'0.1-0.2':<15}")
    print(f"{'Streaming':<25} {'Non':<15} {'Oui':<15}")
    print(f"{'Multilangue':<25} {'Oui':<15} {'Oui':<15}")
    print(f"{'Précision (FR)':<25} {'85-90%':<15} {'92-95%':<15}")
    print(f"{'Coût':<25} {'Gratuit':<15} {'0.0015$/min':<15}")
    print(f"{'Internet requis':<25} {'Non':<15} {'Oui':<15}")
    print("-" * 55)
    
    print("\n💡 Recommandations:")
    print("✅ Azure Speech pour : production, temps réel, multilangue")
    print("✅ Whisper pour : développement, offline, coût zéro")

async def main():
    """Fonction principale"""
    print("🚀 Configuration Azure Speech Services pour SuperWhisper V6")
    print("=" * 60)
    
    # 1. Installation dépendances
    if not install_dependencies():
        print("❌ Installation des dépendances échouée")
        return
    
    # 2. Configuration credentials
    speech_key, speech_region = configure_azure_credentials()
    
    if not speech_key:
        print("⚠️ Configuration Azure non complète")
        print("💡 Configurez les credentials puis relancez le script")
        create_example_config()
        show_performance_comparison()
        return
    
    # 3. Test du backend
    success = await test_azure_speech()
    
    if success:
        print("\n✅ Configuration Azure Speech réussie!")
        print("\n📋 Prochaines étapes:")
        print("1. Intégrez Azure Speech dans votre pipeline SuperWhisper V6")
        print("2. Configurez les langues et modèles selon vos besoins")
        print("3. Ajustez les timeouts pour votre cas d'usage")
        
        create_example_config()
    else:
        print("\n❌ Configuration échouée")
        print("💡 Vérifiez vos credentials Azure et la connectivité")
    
    show_performance_comparison()

if __name__ == "__main__":
    asyncio.run(main()) 