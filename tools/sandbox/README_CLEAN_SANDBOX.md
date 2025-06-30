# 🧹 **Documentation - Système de Purge Sandbox SuperWhisper V6**

## 📋 **Vue d'Ensemble**

Le système de purge sandbox maintient automatiquement la propreté du dépôt en gérant les fichiers temporaires et exploratoires créés par Cursor et les développeurs.

## 🏗️ **Architecture des Solutions**

### **Solution 1 : Répertoire Tampon Cursor**
```
.cursor-tmp/              # Fichiers temporaires Cursor (ignorés Git)
├── scratch_*.py         # Tests jetables auto-générés
├── tmp_*.py            # Fichiers temporaires
└── temp_*.py           # Expérimentations rapides
```

### **Solution 2 : Structure Sandbox Organisée**
```
tests/
├── unit/               # ✅ Tests stables validés (permanents)
├── integration/        # ✅ Tests intégration validés (permanents)
└── sandbox/           # ⏳ Tests exploratoires (≤ 7 jours, purge auto)

PIPELINE/tests/
├── unit/              # ✅ Tests pipeline unitaires stables
├── integration/       # ✅ Tests pipeline intégration stables
└── sandbox/          # ⏳ Tests pipeline exploratoires (≤ 7 jours)
```

## 🔧 **Configuration Automatique**

### **Cursor Settings (`.cursor/settings.json`)**
```json
{
  "cursor.tmpDir": ".cursor-tmp",
  "files.exclude": {
    ".cursor-tmp/**": true,
    "**/*.tmp": true,
    "**/scratch_*.py": true,
    "**/tmp_*.py": true
  }
}
```

### **Git Ignore (`.gitignore`)**
```gitignore
# Fichiers temporaires Cursor - Solution 1 Propreté Répertoire
.cursor-tmp/
scratch_*.py
tmp_*.py
test_scratch_*.py
temp_*.py
```

## 🛠️ **Utilisation du Script de Purge**

### **Commandes Principales**
```bash
# Purge automatique (7 jours par défaut)
python tools/clean_sandbox.py

# Simulation sans suppression (recommandé avant purge)
python tools/clean_sandbox.py --dry-run

# Seuil personnalisé (14 jours)
python tools/clean_sandbox.py --days=14

# Purge forcée sans confirmation
python tools/clean_sandbox.py --force
```

### **Répertoires Surveillés**
- `tests/sandbox/` - Tests exploratoires généraux
- `PIPELINE/tests/sandbox/` - Tests pipeline exploratoires  
- `.cursor-tmp/` - Fichiers temporaires Cursor

### **Extensions Nettoyées**
- `*.py` - Scripts Python temporaires
- `*.json` - Rapports et configs temporaires
- `*.log` - Logs de tests
- `*.wav`, `*.mp3` - Fichiers audio tests
- `*.tmp`, `*.cache` - Fichiers cache
- `*.pkl`, `*.pt`, `*.pth` - Modèles temporaires

## 📊 **Workflow Recommandé**

### **Développement Tests**
1. **Exploration** : Créer tests dans `sandbox/`
2. **Validation** : Tester et itérer rapidement
3. **Promotion** : Déplacer vers `unit/` ou `integration/` si validé
4. **Purge auto** : Fichiers anciens supprimés automatiquement

### **Convention Naming**
```python
# Tests sandbox (temporaires)
test_scratch_stt_experiment.py      # Expérimentation STT
test_pipeline_quick_validation.py   # Validation rapide
tmp_gpu_memory_debug.py             # Debug temporaire

# Tests stables (permanents)
test_stt_optimization.py            # Test STT validé
test_pipeline_integration.py        # Test intégration stable
```

## 🚀 **Automatisation**

### **Tâche Programmée (Windows)**
```batch
# Créer tâche Windows Task Scheduler
schtasks /create /tn "SuperWhisper_SandboxCleanup" /tr "python C:\Dev\SuperWhisper_V6\tools\clean_sandbox.py --force" /sc weekly /d SAT /st 02:00
```

### **Cron Job (Linux/macOS)**
```bash
# Ajout à crontab pour exécution hebdomadaire
0 2 * * 6 cd /path/to/SuperWhisper_V6 && python tools/clean_sandbox.py --force
```

### **Pre-commit Hook (Git)**
```bash
# .git/hooks/pre-commit
#!/bin/bash
echo "🧹 Running sandbox cleanup..."
python tools/clean_sandbox.py --dry-run --days=14
```

## 📈 **Métriques et Monitoring**

### **Exemple de Sortie**
```
🧹 SuperWhisper V6 - Sandbox Cleanup Tool
==================================================
📅 Threshold: 7 days
🔍 Mode: LIVE
⚡ Force: No

📂 Scanning: tests/sandbox
📊 Found 8 files older than 7 days:
   📄 tests/sandbox/test_scratch_gpu_debug.py (12.3 days, 4,521 bytes)
   📄 tests/sandbox/tmp_tts_experiment.py (9.8 days, 2,134 bytes)
   ... and 6 more files

🤔 Delete 8 files? [y/N]: y
🗑️  Deleted: tests/sandbox/test_scratch_gpu_debug.py (4,521 bytes)
🗑️  Deleted: tests/sandbox/tmp_tts_experiment.py (2,134 bytes)

==================================================
📊 CLEANUP SUMMARY
==================================================
🗑️  Files cleaned: 8
💾 Space freed: 45,291 bytes (0.04 MB)

💡 SUGGESTIONS:
   - Consider moving valuable tests to tests/unit/ or tests/integration/
   - Run this script weekly via cron job or task scheduler
```

## 🎯 **Avantages pour SuperWhisper V6**

### **Productivité**
- ✅ **Exploration libre** : Tests rapides sans souci de pollution
- ✅ **Structure claire** : Séparation exploratoire vs stable
- ✅ **Automatisation** : Maintenance zéro après setup

### **Qualité Code**
- ✅ **Dépôt propre** : Git history non pollué
- ✅ **Tests organisés** : Hiérarchie claire par stabilité
- ✅ **Performance** : Recherches/indexation plus rapides

### **Collaboration**
- ✅ **Standards équipe** : Workflow uniforme
- ✅ **Code review** : Focus sur tests permanents
- ✅ **Onboarding** : Structure compréhensible

## 🚨 **Configuration GPU Intégrée**

Le script `clean_sandbox.py` respecte les standards RTX 3090 :
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 exclusif
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable
```

## 🔧 **Maintenance**

### **Vérification Weekly**
```bash
# Vérifier l'état sandbox
python tools/clean_sandbox.py --dry-run

# Statistiques taille répertoires
du -sh tests/sandbox/ PIPELINE/tests/sandbox/ .cursor-tmp/
```

### **Ajustements Configuration**
- Modifier `DEFAULT_THRESHOLD_DAYS` dans `clean_sandbox.py`
- Ajouter nouveaux répertoires dans `SANDBOX_DIRECTORIES`
- Étendre `EXTENSIONS_TO_CLEAN` si nécessaire

---

*Documentation Solutions 1+2 - Propreté Répertoire SuperWhisper V6*  
*Date : 14 Juin 2025 - Système de Purge Automatique Implémenté* 