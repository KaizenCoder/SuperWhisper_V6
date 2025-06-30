# 🚀 Guide de Migration Azure Speech Services - SuperWhisper V6

## 📋 Vue d'Ensemble

Ce guide vous aide à migrer SuperWhisper V6 vers **Azure Speech Services** pour bénéficier de :
- **Latence ultra-faible** (<300ms vs 1-3s avec Whisper)
- **Streaming temps réel** avec reconnaissance continue
- **Qualité supérieure** (92-95% précision vs 85-90%)
- **Support multilangue** avancé

## 🛠️ Installation

### 1. Dépendances
```bash
pip install azure-cognitiveservices-speech>=1.34.0
```

### 2. Configuration Azure
```bash
# Créer ressource Speech dans le portail Azure
# https://portal.azure.com -> Créer une ressource -> Speech Services

# Configurer les variables d'environnement
export AZURE_SPEECH_KEY='votre-clé-azure'
export AZURE_SPEECH_REGION='francecentral'
```

### 3. Test automatique
```bash
python scripts/setup_azure_speech.py
```

## ⚙️ Configuration SuperWhisper V6

### Configuration STT avec Azure Speech

```python
# config/stt_config.py
STT_CONFIG = {
    'preferred_backend': 'azure_speech',
    'backends': {
        'azure_speech': {
            'enabled': True,
            'azure_speech_key': os.getenv('AZURE_SPEECH_KEY'),
            'azure_speech_region': 'francecentral',
            'language': 'fr-FR',
            'continuous_recognition': True,
            'enable_detailed_results': True,
            'enable_word_level_timestamps': True,
            'segmentation_silence_timeout_ms': 500,
            'initial_silence_timeout_ms': 5000
        }
    }
}
```

## 🔄 Utilisation

### Mode Streaming (Recommandé)

```python
from STT.backends.azure_speech_backend import AzureSpeechBackend

async def streaming_example():
    # Configuration
    config = {
        'azure_speech_key': os.getenv('AZURE_SPEECH_KEY'),
        'azure_speech_region': 'francecentral',
        'language': 'fr-FR'
    }
    
    # Créer backend
    backend = AzureSpeechBackend(config)
    
    # Callbacks streaming
    def on_interim(text):
        print(f"🎤 Interim: {text}")
    
    def on_final(result):
        print(f"✅ Final: {result.text} (confiance: {result.confidence:.2f})")
    
    # Démarrer reconnaissance continue
    await backend.start_continuous_recognition(
        interim_callback=on_interim,
        final_callback=on_final
    )
    
    # Push audio en temps réel
    while recording:
        audio_chunk = get_audio_chunk()  # Votre fonction
        await backend.push_audio(audio_chunk)
    
    # Arrêter
    await backend.stop_continuous_recognition()
```

### Mode Transcription Simple

```python
async def transcription_example():
    backend = AzureSpeechBackend(config)
    
    # Charger audio
    audio = load_audio_file("recording.wav")  # numpy array
    
    # Transcrire
    result = await backend.transcribe(audio)
    
    print(f"Texte: {result.text}")
    print(f"Confiance: {result.confidence}")
    print(f"Segments: {result.segments}")
    print(f"Temps: {result.processing_time:.2f}s")
```

## 🔧 Optimisations Performance

### Latence Ultra-faible
```python
config = {
    'segmentation_silence_timeout_ms': 300,  # Plus réactif
    'initial_silence_timeout_ms': 3000,      # Démarrage rapide
    'enable_detailed_results': False,        # Si timestamps non requis
}
```

### Qualité Maximale
```python
config = {
    'enable_detailed_results': True,
    'enable_word_level_timestamps': True,
    'profanity_option': 'Masked',
    'custom_endpoint_id': 'votre-modele-custom'  # Si modèle personnalisé
}
```

## 🌍 Support Multilangue

### Configuration Multi-langues
```python
# Détection automatique
from azure.cognitiveservices.speech import AutoDetectSourceLanguageConfig

language_config = AutoDetectSourceLanguageConfig(
    languages=["fr-FR", "en-US", "es-ES"]
)

# Ou langue fixe
config['language'] = 'en-US'  # Anglais
config['language'] = 'es-ES'  # Espagnol
config['language'] = 'de-DE'  # Allemand
```

## 📊 Comparaison Performance

| Critère | Whisper Local | Azure Speech | Amélioration |
|---------|---------------|--------------|--------------|
| Latence | 1-3s | <300ms | **10x plus rapide** |
| RTF | 0.3-1.0 | 0.1-0.2 | **5x plus efficace** |
| Streaming | ❌ | ✅ | **Temps réel** |
| Précision (FR) | 85-90% | 92-95% | **+7% précision** |
| Word timestamps | ⚠️ | ✅ | **Natif** |

## 💰 Coûts

### Pricing Azure Speech
- **Reconnaissance standard** : 0.0015$ par minute
- **Custom Speech** : 0.0024$ par minute
- **Tier gratuit** : 5h/mois incluses

### Calcul pour SuperWhisper V6
```python
# Exemple : 2h d'usage/jour
minutes_par_mois = 2 * 60 * 30  # 3600 min
cout_mensuel = 3600 * 0.0015    # 5.4$ par mois
```

## 🔐 Sécurité et Conformité

### Données
- **Chiffrement** en transit et au repos
- **Pas de stockage** des données audio (streaming)
- **Conformité** RGPD, HIPAA, SOC 2

### Région France
```python
config['azure_speech_region'] = 'francecentral'  # Données en France
```

## 🚨 Migration depuis Whisper

### 1. Backup Actuel
```bash
# Sauvegarder configuration existante
cp STT/config/stt_config.py STT/config/stt_config_backup.py
```

### 2. Migration Progressive
```python
# Dual backend pour transition
STT_CONFIG = {
    'backends': {
        'azure_speech': {
            'enabled': True,
            'priority': 10  # Priorité haute
        },
        'whisper': {
            'enabled': True,
            'priority': 5   # Fallback
        }
    }
}
```

### 3. A/B Testing
```python
async def test_both_backends():
    # Test parallèle
    azure_result = await azure_backend.transcribe(audio)
    whisper_result = await whisper_backend.transcribe(audio)
    
    print(f"Azure: {azure_result.text} ({azure_result.processing_time:.2f}s)")
    print(f"Whisper: {whisper_result.text} ({whisper_result.processing_time:.2f}s)")
```

## 🐛 Debugging

### Logs Détaillés
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Activer logs Azure Speech
config['debug'] = True
```

### Health Checks
```python
# Vérifier état du backend
if backend.health_check():
    print("✅ Azure Speech opérationnel")
else:
    print("❌ Problème de connexion")
```

### Métriques
```python
# Statistiques backend
stats = backend.get_stats()
print(f"Requêtes totales: {stats['total_requests']}")
print(f"Temps moyen: {stats['avg_processing_time']:.2f}s")
print(f"Taux d'erreur: {stats['error_rate']:.1%}")
```

## 🚀 Prochaines Étapes

1. **Tester** avec le script de configuration
2. **Configurer** vos credentials Azure
3. **Intégrer** dans votre pipeline SuperWhisper V6
4. **Optimiser** les paramètres selon votre usage
5. **Monitorer** les performances en production

## 📞 Support

- **Documentation Azure** : [docs.microsoft.com/azure/cognitive-services/speech-service](https://docs.microsoft.com/azure/cognitive-services/speech-service)
- **Exemples Python** : [github.com/Azure-Samples/cognitive-services-speech-sdk](https://github.com/Azure-Samples/cognitive-services-speech-sdk)
- **Support Azure** : Via le portail Azure

---

**🎉 Avec Azure Speech Services, SuperWhisper V6 atteint un nouveau niveau de performance pour la reconnaissance vocale temps réel !** 