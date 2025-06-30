# 🧪 Testing - Suite Complète de Tests

> **Tous les outils pour tester et valider votre système SuperWhisper !**

---

## 🎯 **Outils Principaux**

### 🤖 **Assistants de Test**

#### [`run_assistant.py`](run_assistant.py) - 12KB
**Assistant principal de test avec interface complète**
```bash
python testing/run_assistant.py
```
- ✅ Interface utilisateur avancée
- ✅ Tests automatisés complets
- ✅ Rapports détaillés
- ✅ Support STT/TTS

#### [`run_assistant_simple.py`](run_assistant_simple.py) - 5KB
**Assistant simple pour tests rapides**
```bash
python testing/run_assistant_simple.py
```
- ✅ Tests de base ultra-rapides
- ✅ Interface minimaliste
- ✅ Idéal pour debug

#### [`run_assistant_coqui.py`](run_assistant_coqui.py) - 5KB
**Assistant spécialisé Coqui TTS**
```bash
python testing/run_assistant_coqui.py
```
- ✅ Tests spécifiques Coqui
- ✅ Validation modèles TTS
- ✅ Performance vocale

### 🔬 **Suite Complète**

#### [`run_complete_tests.py`](run_complete_tests.py) - 16KB
**Suite complète de tous les tests système**
```bash
python testing/run_complete_tests.py
```
- ✅ Tests GPU RTX 3090
- ✅ Pipeline complet STT→TTS
- ✅ Validation performance
- ✅ Rapport final détaillé

---

## 🎤 **Tests STT/TTS Spécialisés** (Sous-répertoire `stt/`)

### [`test_stt_validation_individual.py`](stt/test_stt_validation_individual.py) - 16KB
**Tests STT individuels par modèle**
```bash
python testing/stt/test_stt_validation_individual.py
```
- ✅ Test chaque modèle STT séparément
- ✅ Métriques précision par modèle
- ✅ Rapport comparatif

### [`test_stt_validation_transmission.py`](stt/test_stt_validation_transmission.py) - 18KB
**Tests transmission temps réel STT**
```bash
python testing/stt/test_stt_validation_transmission.py
```
- ✅ Pipeline audio temps réel
- ✅ Latence et performance
- ✅ Validation transmission

### [`test_tts_validation_transmission.py`](stt/test_tts_validation_transmission.py) - 8KB
**Tests transmission temps réel TTS**
```bash
python testing/stt/test_tts_validation_transmission.py
```
- ✅ Génération voix temps réel
- ✅ Qualité audio
- ✅ Performance GPU

---

## 🚀 **Utilisation par Scénario**

### ⚡ **Test Rapide (2 minutes)**
```bash
python testing/run_assistant_simple.py
```

### 🔧 **Test Développement (5 minutes)**
```bash
python testing/run_assistant.py
```

### 🎯 **Test Complet (15 minutes)**
```bash
python testing/run_complete_tests.py
```

### 🎤 **Tests Vocal Spécialisés**
```bash
# Tests STT individuels
python testing/stt/test_stt_validation_individual.py

# Tests transmission STT
python testing/stt/test_stt_validation_transmission.py

# Tests transmission TTS
python testing/stt/test_tts_validation_transmission.py
```

---

## 📊 **Ordre de Test Recommandé**

### 🔥 **Première Installation**
1. `run_assistant_simple.py` - Validation base
2. `run_assistant.py` - Test principal
3. `run_complete_tests.py` - Validation complète

### 🔄 **Tests Quotidiens**
1. `run_assistant_simple.py` - Check rapide
2. Tests STT/TTS si modifications vocales

### 🚀 **Avant Release**
1. `run_complete_tests.py` - Suite complète
2. Tous les tests STT/TTS
3. Validation performance

---

## 📈 **Statistiques & Performance**

### 📊 **Temps d'Exécution Moyens**
- **Simple** : 1-2 minutes
- **Assistant** : 3-5 minutes
- **Complet** : 10-15 minutes
- **STT Individual** : 5-8 minutes
- **STT/TTS Transmission** : 8-12 minutes

### 🎯 **Couverture Tests**
- **GPU RTX 3090** : ✅ 100%
- **Pipeline STT** : ✅ 100%
- **Pipeline TTS** : ✅ 100%
- **Transmission** : ✅ 100%
- **Performance** : ✅ 100%

---

## 🔗 **Liens Utiles**
- [🎯 Index Principal](../INDEX_OUTILS_COMPLET.md)
- [📊 Monitoring](../monitoring/README.md)
- [🧠 Analyse Mémoire](../memory/README.md)
- [🚀 Guide Démarrage](../GUIDE_DEMARRAGE_RAPIDE_OUTILS.md)

---

*Testing SuperWhisper V6 - Validation Complète* 🧪 