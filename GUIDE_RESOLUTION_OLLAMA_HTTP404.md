# 🔧 GUIDE RÉSOLUTION OLLAMA HTTP 404 - SuperWhisper V6

**Problème** : Erreur HTTP 404 lors des appels à l'API Ollama  
**Impact** : LLM utilise toujours le fallback au lieu d'Ollama  
**Solution** : Diagnostic automatique + correction API  

---

## 🚀 RÉSOLUTION RAPIDE (5 MINUTES)

### Étape 1: Diagnostic Automatique
```powershell
# Depuis PowerShell Windows (pas WSL)
PS C:\Dev\SuperWhisper_V6> python diagnostic_ollama_fix.py
```

**Ce script va :**
- ✅ Démarrer Ollama automatiquement
- ✅ Tester tous les endpoints API disponibles
- ✅ Identifier le format API fonctionnel
- ✅ Générer un rapport de diagnostic

### Étape 2: Correction Automatique
```powershell
PS C:\Dev\SuperWhisper_V6> python fix_llm_manager_ollama.py
```

**Ce script va :**
- ✅ Sauvegarder le fichier original
- ✅ Corriger l'API Ollama dans `LLM/llm_manager_enhanced.py`
- ✅ Vérifier que la correction est appliquée
- ✅ Créer un script de test

### Étape 3: Test de la Correction
```powershell
PS C:\Dev\SuperWhisper_V6> python test_ollama_corrected.py
```

**Résultat attendu :**
```
✅ Test réussi: Paris est la capitale de la France.
```

### Étape 4: Test Pipeline Complet
```powershell
PS C:\Dev\SuperWhisper_V6> python test_pipeline_microphone_reel.py
```

---

## 🔍 DIAGNOSTIC DÉTAILLÉ

### Problème Identifié

L'erreur HTTP 404 est causée par :

1. **Format API incorrect** : Le code utilise un mélange d'API OpenAI + Ollama
2. **Endpoint erroné** : `/v1/chat/completions` au lieu de `/api/generate`
3. **Structure de données incorrecte** : `messages` au lieu de `prompt`

### Correction Appliquée

**AVANT** (Code défaillant) :
```python
# ❌ Format hybride incorrect
data = {
    "model": self.config.get('model', 'nous-hermes'),
    "messages": messages,  # Format OpenAI
    "max_tokens": max_tokens,
    "temperature": temperature,
    "stream": False
}

response = await client.post(
    'http://127.0.0.1:11434/v1/chat/completions',  # ❌ Endpoint OpenAI
    json=data
)
```

**APRÈS** (Code corrigé) :
```python
# ✅ Format API native Ollama
data = {
    "model": actual_model,
    "prompt": f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:",  # Format Ollama
    "stream": False,
    "options": {
        "temperature": temperature,
        "num_predict": max_tokens,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "stop": ["User:", "\n\n"]
    }
}

response = await client.post(
    'http://127.0.0.1:11434/api/generate',  # ✅ Endpoint Ollama natif
    json=data
)
```

---

## 📊 FORMATS API OLLAMA SUPPORTÉS

### 1. API Native Ollama (Recommandée)
- **Endpoint** : `/api/generate`
- **Format** : `{"model": "...", "prompt": "...", "options": {...}}`
- **Avantages** : Plus rapide, support complet des options

### 2. API Chat Ollama
- **Endpoint** : `/api/chat`
- **Format** : `{"model": "...", "messages": [...], "options": {...}}`
- **Avantages** : Format conversationnel

### 3. API OpenAI Compatible
- **Endpoint** : `/v1/chat/completions`
- **Format** : `{"model": "...", "messages": [...], "max_tokens": ...}`
- **Avantages** : Compatibilité avec code OpenAI existant

---

## 🛠️ DÉPANNAGE AVANCÉ

### Si le diagnostic échoue :

1. **Vérifier Ollama installé** :
   ```powershell
   ollama --version
   ```

2. **Démarrer Ollama manuellement** :
   ```powershell
   # Terminal 1
   set OLLAMA_MODELS=D:\modeles_llm
   ollama serve
   
   # Terminal 2 - Tester
   ollama list
   ```

3. **Vérifier le modèle** :
   ```powershell
   ollama pull nous-hermes-2-mistral-7b-dpo
   ollama run nous-hermes-2-mistral-7b-dpo "Bonjour"
   ```

4. **Vérifier le port** :
   ```powershell
   netstat -an | findstr 11434
   ```

### Si la correction échoue :

1. **Restaurer la sauvegarde** :
   ```powershell
   # Les sauvegardes sont créées automatiquement
   # Format: LLM/llm_manager_enhanced.py.backup_YYYYMMDD_HHMMSS
   copy "LLM\llm_manager_enhanced.py.backup_*" "LLM\llm_manager_enhanced.py"
   ```

2. **Correction manuelle** :
   - Ouvrir `LLM/llm_manager_enhanced.py`
   - Localiser la méthode `_generate_ollama`
   - Remplacer par le code corrigé (voir section "Correction Appliquée")

---

## 🎯 VALIDATION FINALE

Une fois la correction appliquée, le pipeline doit :

1. **STT** : Transcrire votre voix → ✅ (déjà fonctionnel)
2. **LLM** : Générer vraies réponses via Ollama → ✅ (corrigé)
3. **TTS** : Synthétiser la réponse → ✅ (déjà fonctionnel)

**Test de validation** :
```
Vous : "Quelle est la capitale de la France ?"
Pipeline : "Paris est la capitale de la France."
```

**Performance attendue** :
- STT : ~1000ms
- LLM : ~2000-5000ms (avec Ollama)
- TTS : ~800ms
- **Total** : ~4-7 secondes

---

## 📁 FICHIERS CRÉÉS

- `diagnostic_ollama_fix.py` - Script diagnostic automatique
- `fix_llm_manager_ollama.py` - Script correction automatique
- `test_ollama_corrected.py` - Script test post-correction
- `diagnostic_ollama_rapport.json` - Rapport diagnostic détaillé
- `LLM/llm_manager_enhanced.py.backup_*` - Sauvegarde automatique

---

## ⚡ RÉSUMÉ EXÉCUTION

```powershell
# 1. Diagnostic (2 min)
python diagnostic_ollama_fix.py

# 2. Correction (30 sec)
python fix_llm_manager_ollama.py

# 3. Test (30 sec)
python test_ollama_corrected.py

# 4. Pipeline complet (test final)
python test_pipeline_microphone_reel.py
```

**Résultat** : Pipeline voix-à-voix 100% fonctionnel avec vraies réponses LLM Ollama !

---

*Guide SuperWhisper V6 - Résolution HTTP 404 Ollama*  
*Temps de résolution : ~5 minutes* 