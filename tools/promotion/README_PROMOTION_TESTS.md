# 🚀 **GUIDE PROMOTION TESTS - SUPERWHISPER V6**

## 📋 **WORKFLOW COMPLET DE PROMOTION**

### **🎯 CONTEXTE : TESTS DANS RÉPERTOIRES "."**

#### **Question Initiale**
> "Si test validé dans un répertoire commençant par un point comment promouvoir ce test?"

#### **Réponse : 3 Scénarios de Promotion**

### **🔄 SCÉNARIOS DE PROMOTION**

#### **1️⃣ Promotion Standard (Recommandée)**
```
tests/sandbox/test_experiment.py → tests/unit/test_experiment.py
PIPELINE/tests/sandbox/pipeline_test.py → PIPELINE/tests/unit/pipeline_test.py
```

#### **2️⃣ Promotion depuis .cursor-tmp/ (Récupération)**
```
.cursor-tmp/scratch_stt_test.py → tests/unit/test_stt_recovered.py
.cursor-tmp/temp_debug.py → tests/integration/test_debug_validated.py
```

#### **3️⃣ Promotion Cross-Répertoires**
```
tests/sandbox/pipeline_specific.py → PIPELINE/tests/unit/test_pipeline_specific.py
.cursor-tmp/scratch_pipeline.py → PIPELINE/tests/integration/test_pipeline_recovered.py
```

---

## 🛠️ **OUTIL DE PROMOTION : `tools/promote_test.py`**

### **✅ DÉMONSTRATION RÉUSSIE (15/06/2025 00:15)**

**Test effectué :**
```bash
python tools/promote_test.py tests/sandbox/test_scratch_example.py tests/unit/ --copy --force
```

**Résultat :**
- ✅ **PROMOTION RÉUSSIE** - Fichier promu disponible
- ✅ **Header mis à jour** - Métadonnées promotion ajoutées
- ✅ **Mode copie** - Original préservé dans sandbox
- ✅ **Configuration GPU** - Standards RTX 3090 appliqués

---

## 📋 **SYNTAXE COMPLÈTE**

### **Usage Basique**
```bash
python tools/promote_test.py <source> <target_dir> [options]
```

### **Exemples Pratiques**

#### **Promotion Simple**
```bash
# Depuis sandbox vers unit (déplacement)
python tools/promote_test.py tests/sandbox/test_experiment.py tests/unit/

# Depuis .cursor-tmp vers unit (déplacement)
python tools/promote_test.py .cursor-tmp/scratch_stt.py tests/unit/
```

#### **Promotion avec Options**
```bash
# Copie (garder original)
python tools/promote_test.py tests/sandbox/test_experiment.py tests/unit/ --copy

# Renommage personnalisé
python tools/promote_test.py .cursor-tmp/temp_debug.py tests/unit/ --rename=test_gpu_validation.py

# Mode automatique (sans confirmation)
python tools/promote_test.py tests/sandbox/quick_test.py tests/integration/ --force

# Combinaison multiple
python tools/promote_test.py .cursor-tmp/scratch_pipeline.py PIPELINE/tests/unit/ --copy --rename=test_pipeline_optimized.py --force
```

#### **Vers Répertoires PIPELINE**
```bash
# Promotion vers PIPELINE/tests/unit/
python tools/promote_test.py tests/sandbox/pipeline_test.py PIPELINE/tests/unit/

# Promotion vers PIPELINE/tests/integration/
python tools/promote_test.py .cursor-tmp/scratch_pipeline.py PIPELINE/tests/integration/
```

---

## 🎯 **RÉPERTOIRES CIBLES AUTORISÉS**

### **Standard**
- `tests/unit/` - Tests unitaires stables
- `tests/integration/` - Tests intégration stables
- `PIPELINE/tests/unit/` - Tests unitaires pipeline
- `PIPELINE/tests/integration/` - Tests intégration pipeline

### **Validation Sécurisée**
- ✅ L'outil valide les répertoires cibles
- ⚠️ Avertissement pour répertoires non-standard
- 🤔 Confirmation utilisateur requise si hors-standard

---

## 🔧 **FONCTIONNALITÉS INTELLIGENTES**

### **🧹 Nettoyage Automatique Noms**
```python
# Avant promotion (noms temporaires)
.cursor-tmp/scratch_stt_test.py → tests/unit/test_stt_test.py
.cursor-tmp/tmp_debug_gpu.py → tests/unit/test_debug_gpu.py
tests/sandbox/temp_experiment.py → tests/unit/test_experiment.py
```

### **📝 Header Automatique**
```python
#!/usr/bin/env python3
"""
✅ Test Promu - SuperWhisper V6

Promu depuis: tests/sandbox/test_scratch_example.py
Vers: tests/unit/test_scratch_example.py
Date promotion: 15/06/2025 00:15
Statut: Test validé et permanent

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""
```

### **🛡️ Protection Anti-Écrasement**
```python
# Si test_example.py existe déjà
tests/unit/test_example.py → Existe
tests/unit/test_example_01.py → Créé automatiquement
```

---

## 📊 **WORKFLOW COMPLET**

### **Étape 1 : Identifier Test à Promouvoir**
```bash
# Lister tests candidats
ls tests/sandbox/*.py
ls .cursor-tmp/*.py
ls PIPELINE/tests/sandbox/*.py
```

### **Étape 2 : Promouvoir avec Outil**
```bash
# Promotion avec confirmation
python tools/promote_test.py <source> <target>

# Promotion automatique
python tools/promote_test.py <source> <target> --force
```

### **Étape 3 : Finaliser avec Git**
```bash
# Ajouter test promu
git add tests/unit/test_promoted.py

# Si mode déplacement (pas copie)
git rm tests/sandbox/test_original.py

# Commit
git commit -m "feat(tests): promote validated test

- Moved from sandbox/temp to stable unit tests
- Test validated and ready for CI/CD
- RTX 3090 configuration applied"
```

### **Étape 4 : Validation Finale**
```bash
# Tester le fichier promu
python tests/unit/test_promoted.py

# Tests avec pytest
pytest tests/unit/test_promoted.py -v

# CI/CD integration
pytest tests/unit/ -v
```

---

## 🎊 **AVANTAGES PROMOTION AUTOMATISÉE**

### **✅ Bénéfices**
- **🔄 Traçabilité** : Métadonnées promotion complètes
- **🛡️ Sécurité** : Validation cibles + anti-écrasement
- **🧹 Propreté** : Nettoyage noms automatique
- **📋 Standards** : Configuration GPU RTX 3090 appliquée
- **⚡ Rapidité** : Promotion en une commande
- **🎯 Flexibilité** : Copie ou déplacement au choix

### **🔧 Cas d'Usage**
1. **Récupération urgente** : Fichier important dans `.cursor-tmp/`
2. **Validation sandbox** : Test exploratoire confirmé stable
3. **Réorganisation** : Migration vers structure pipeline
4. **Nettoyage périodique** : Promotion avant purge automatique

---

## 📈 **MÉTRIQUES & VALIDATION**

### **Test de Validation (15/06/2025)**
- ✅ **Source** : `tests/sandbox/test_scratch_example.py`
- ✅ **Cible** : `tests/unit/test_scratch_example.py`
- ✅ **Mode** : Copie (original préservé)
- ✅ **Header** : Mis à jour automatiquement
- ✅ **GPU Config** : RTX 3090 appliquée
- ✅ **Durée** : < 3 secondes

### **Répertoires Supportés (4)**
- `tests/unit/` ✅
- `tests/integration/` ✅
- `PIPELINE/tests/unit/` ✅
- `PIPELINE/tests/integration/` ✅

---

## 💡 **CONSEILS PRATIQUES**

### **🎯 Quand Promouvoir**
- ✅ Test validé manuellement
- ✅ Logique stable et réutilisable
- ✅ Performance acceptable
- ✅ Documentation suffisante

### **🚨 Quand NE PAS Promouvoir**
- ❌ Test encore expérimental
- ❌ Code temporaire/debug
- ❌ Données hardcodées spécifiques
- ❌ Dépendances externes non-stables

### **📋 Checklist Promotion**
1. [ ] Test validé fonctionnellement
2. [ ] Code propre et documenté
3. [ ] Configuration GPU RTX 3090 appliquée
4. [ ] Pas de dépendances temporaires
5. [ ] Nom de fichier explicite
6. [ ] Répertoire cible approprié

---

**Documentation Promotion Tests - SuperWhisper V6**  
*Mise à jour : 15/06/2025 00:20*  
*Statut : ✅ Opérationnel et Validé* 