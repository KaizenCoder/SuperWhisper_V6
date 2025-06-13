# 🛠️ **OUTIL GÉNÉRATEUR AIDE EXTERNE - SUPERWHISPER V6**

## 🎯 **OBJECTIF**

Outil automatisé pour créer des demandes d'aide externe optimisées :
- ✅ **Un seul fichier .md** (vs 71 fichiers ZIP)
- ✅ **Code essentiel agrégé** et lisible
- ✅ **Compatible consultants externes** (markdown natif)
- ✅ **Taille optimale** (<50KB vs 244KB)

---

## 🚀 **UTILISATION**

### **1. Utilisation CLI**

```bash
# Exemple validation microphone
python tools/generateur_aide_externe.py \
    --probleme "Validation Microphone Live Phase 4 STT" \
    --fichiers STT/unified_stt_manager.py STT/backends/prism_stt_backend.py STT/vad_manager.py scripts/validation_microphone_live_equipe.py \
    --contexte "Architecture STT parfaite sur fichiers, échec total microphone streaming" \
    --urgence CRITIQUE \
    --titre VALIDATION_MICROPHONE

# Exemple performance TTS
python tools/generateur_aide_externe.py \
    --probleme "Optimisation Performance TTS" \
    --fichiers TTS/tts_manager.py config/tts_config.yaml \
    --urgence ÉLEVÉE \
    --titre PERFORMANCE_TTS
```

### **2. Utilisation Python**

```python
from tools.generateur_aide_externe import GenerateurAideExterne

generator = GenerateurAideExterne()

# Création aide externe
resultat = generator.creer_aide_externe(
    probleme="Validation Microphone Live Phase 4 STT",
    fichiers_critiques=[
        "STT/unified_stt_manager.py",
        "STT/backends/prism_stt_backend.py", 
        "STT/vad_manager.py",
        "scripts/validation_microphone_live_equipe.py"
    ],
    contexte="Architecture STT parfaite sur fichiers, échec total microphone",
    urgence="CRITIQUE",
    titre_court="VALIDATION_MICROPHONE"
)

print(f"Document créé : {resultat['principal']}")
```

---

## 📋 **PARAMÈTRES**

### **Arguments Obligatoires**
- `--probleme` : Description claire du problème (ex: "Validation Microphone Live")
- `--fichiers` : Liste fichiers critiques à analyser (séparés par espaces)

### **Arguments Optionnels**
- `--contexte` : Contexte supplémentaire pour aide externe
- `--urgence` : Niveau urgence (`NORMALE`, `ÉLEVÉE`, `CRITIQUE`)
- `--titre` : Titre court pour nommage fichiers (ex: `VALIDATION_MICROPHONE`)

---

## 📁 **FICHIERS GÉNÉRÉS**

### **Document Principal**
- **Nom** : `{TITRE}_{TIMESTAMP}.md`
- **Contenu** :
  - Contexte problème
  - Code essentiel agrégé des fichiers critiques
  - Analyse zones suspectes
  - Demande aide exhaustive avec contraintes techniques

### **Document Récapitulatif**
- **Nom** : `RECAP_{TITRE}_{TIMESTAMP}.md`
- **Contenu** :
  - Résumé livrable
  - Instructions utilisation
  - Avantages vs approche ZIP

---

## 🔧 **FONCTIONNALITÉS AVANCÉES**

### **Extraction Code Intelligent**
- **Python** : Classes/fonctions principales + docstrings + imports
- **YAML/JSON** : Configuration complète
- **Markdown** : Contenu intégral
- **Autres** : Première partie (2000 chars)

### **Optimisations**
- **Taille limitée** : Code tronqué si >5000 chars
- **Lisibilité** : Commentaires et structure préservés
- **Contexte** : Analyse automatique type fichier

### **Gestion Erreurs**
- **Fichiers manquants** : Signalés mais n'arrêtent pas le process
- **Erreurs lecture** : Documentées dans sortie
- **Validation** : Vérification fichiers avant traitement

---

## 🎯 **CAS D'USAGE TYPES**

### **1. Problème Performance**
```bash
python tools/generateur_aide_externe.py \
    --probleme "Latence excessive TTS Pipeline" \
    --fichiers TTS/tts_manager.py TTS/backends/coqui_backend.py \
    --urgence ÉLEVÉE
```

### **2. Bug Integration**
```bash
python tools/generateur_aide_externe.py \
    --probleme "Échec intégration STT-LLM" \
    --fichiers STT/unified_stt_manager.py LLM/llm_manager.py \
    --contexte "Pipeline fonctionne séparément, échec coordination"
```

### **3. Configuration Hardware**
```bash
python tools/generateur_aide_externe.py \
    --probleme "Configuration GPU RTX 3090" \
    --fichiers config/gpu_config.py scripts/gpu_test.py \
    --urgence CRITIQUE
```

---

## ✅ **AVANTAGES vs PACKAGE ZIP**

| Aspect | Package ZIP (71 fichiers) | Document .md Unique |
|--------|---------------------------|---------------------|
| **Taille** | 244KB | <50KB |
| **Lisibilité** | Nécessite décompression | Lecture directe |
| **Envoi** | Bloqué par email/sécurité | Compatible partout |
| **Navigation** | 71 fichiers à explorer | Structure claire |
| **Réponse** | Complexe intégration | Code direct utilisable |

---

**🚀 OUTIL PRÊT POUR AIDE EXTERNE OPTIMISÉE !** 