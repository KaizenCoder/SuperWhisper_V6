# 📋 PRD - HOMOGÉNISATION DU MAPPING GPU SUPERWHISPER V6

---

**Projet :** Correction et Homogénisation du Mapping GPU SuperWhisper V6  
**Version :** 2.0 [OPTIMISÉE avec Memory Leak V4.0 + Parallélisation]  
**Date :** Juin 2025  
**Priorité :** CRITIQUE  
**Durée estimée :** 12-16 heures (40 fichiers) [64% GAIN vs 33h séquentiel]  
**Durée séquentielle :** 33 heures (baseline de référence)  

---

## 🎯 CONTEXTE ET PROBLÉMATIQUE

### Situation Actuelle
Le projet SuperWhisper V6 présente une **méthodologie de sélection et contrôle GPU non homogène** à travers ses 89 fichiers Python/PowerShell. Cette hétérogénéité génère :

- **Risques de performance** : Utilisation accidentelle de RTX 5060 Ti (16GB) au lieu de RTX 3090 (24GB)
- **Instabilité système** : Mappings GPU incohérents entre modules
- **Maintenance complexe** : Pas de standard unifié pour la sélection GPU
- **Erreurs silencieuses** : Aucune validation systématique du GPU utilisé

### Découverte Critique
L'analyse factuelle révèle que la **configuration GPU nécessite deux variables d'environnement** pour fonctionner correctement :
- `CUDA_VISIBLE_DEVICES='1'` seul ne suffit pas
- `CUDA_DEVICE_ORDER='PCI_BUS_ID'` est **obligatoire** pour respecter l'ordre physique des GPU

### Configuration Matérielle
```
Configuration physique du système :
- GPU Bus PCI 0 : RTX 5060 Ti (16GB) → STRICTEMENT INTERDITE
- GPU Bus PCI 1 : RTX 3090 (24GB) → SEULE GPU AUTORISÉE

⚠️ ATTENTION : Sans CUDA_DEVICE_ORDER='PCI_BUS_ID', PyTorch peut ordonner les GPU différemment !
```

### Impact Business
- **Performance dégradée** sur les tâches IA critiques
- **Risque de plantage** lors de traitement de gros volumes  
- **Incohérence utilisateur** avec des temps de réponse variables
- **Maintenance difficile** due aux standards non homogènes

---

## 🎯 OBJECTIFS

### Objectif Principal
**Homogénéiser et sécuriser la sélection GPU** dans tous les scripts du projet pour garantir l'utilisation exclusive de la RTX 3090.

### Objectifs Spécifiques
1. **Ajouter la configuration GPU complète** dans les 40 scripts identifiés
2. **S'assurer de l'utilisation cohérente de `cuda:0`** dans le code (qui pointera vers RTX 3090)
3. **Implémenter une validation systématique** de sélection GPU
4. **Garantir zéro régression fonctionnelle**
5. **Documenter les standards** pour développements futurs

---

## 🎯 COMPRÉHENSION FACTUELLE CONFIRMÉE

### **Configuration Physique Réelle Validée :**
```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
GPU 0: NVIDIA GeForce RTX 5060 Ti (16311 MiB = ~16GB) ❌ INTERDITE
GPU 1: NVIDIA GeForce RTX 3090 (24576 MiB = ~24GB)    ✅ CIBLE
```

### **Logique CUDA_VISIBLE_DEVICES Confirmée :**
1. **`CUDA_VISIBLE_DEVICES='1'`** = Rendre visible UNIQUEMENT le GPU physique 1 (RTX 3090)
2. **PyTorch remapping automatique** = Le seul GPU visible devient `cuda:0` dans le code
3. **Résultat final** = `cuda:0` dans PyTorch pointe vers RTX 3090 ✅
4. **RTX 5060 Ti devient inaccessible** = Aucun risque d'utilisation accidentelle

### **Validation Obligatoire avec Script de Diagnostic :**
```python
# Utiliser OBLIGATOIREMENT ce script pour validation :
python "C:\Dev\SuperWhisper_V6\test_diagnostic_rtx3090.py"

# Le script DOIT confirmer :
# ✅ CUDA_VISIBLE_DEVICES='1' configuré
# ✅ GPU 0 (après mapping) = RTX 3090 24GB
# ✅ RTX 5060 Ti invisible/inaccessible
# ✅ Configuration fonctionnelle validée
```

### **Points Critiques de Compréhension :**
- **CUDA_VISIBLE_DEVICES='1'** ne change PAS l'ordre, il MASQUE le GPU 0
- **PyTorch voit 1 seul GPU** (RTX 3090) qu'il nomme automatiquement `cuda:0`
- **Le code utilise `cuda:0`** qui pointe maintenant vers RTX 3090
- **Aucune confusion possible** : RTX 5060 Ti est complètement invisible

---

## 🚀 OPTIMISATIONS PERFORMANCE VALIDÉES

### Memory Leak Solution V4.0 - Intégration Obligatoire
- **Script central** : `memory_leak_v4.py` (solution finalisée et validée)
- **Memory leak prevention** : 0% memory leak détecté sur 10/10 stress tests
- **Context manager automatique** : `@gpu_test_cleanup()` pour tous tests GPU
- **Queue GPU exclusive** : Sémaphore multiprocess pour RTX 3090
- **Monitoring temps réel** : Mémoire, fragmentation, performance
- **Emergency recovery** : Reset automatique si memory leak critique
- **Métriques Prometheus** : Monitoring centralisé intégré

### Parallélisation Validée - 64% Gain Performance
```
CONFIGURATION SYSTÈME VALIDÉE :
- RAM : 64GB (32+32GB DDR4-4800) ✅
- CPU : Intel Core Ultra 7 265K (20 threads logiques) ✅
- GPU : RTX 3090 (24GB VRAM) sur Bus PCI 1 ✅
- Memory Leak Solution : 10/10 tests réussis ✅

GAINS PERFORMANCE CONFIRMÉS :
- Phase 1 (Préparation) : 3h → 1.5h (50% gain)
- Phase 2 (13 modules core) : 10h → 3.5h (65% gain) 
- Phase 3 (27 scripts test) : 15h → 4.5h (70% gain)
- Phase 4 (Tests système) : 3h → 3h (séquentiel obligatoire)
- Phase 5 (Documentation) : 2h → 1h (50% gain)

TOTAL : 33h → 13.5h (59% gain validé)
```

### Architecture Technique Parallélisation
- **ThreadPool** : 8-10 workers CPU simultanés optimaux
- **GPU Queue** : Accès RTX 3090 exclusif via sémaphore multiprocess
- **Memory Management** : `memory_leak_v4.py` intégré à chaque worker
- **Git Workflow** : Branches dédiées par worker pour éviter conflits
- **Monitoring centralisé** : Prometheus metrics temps réel
- **Fallback automatique** : Séquentiel si instabilité détectée

### Contraintes et Limitations Parallélisation
- **GPU unique** : RTX 3090 = queue obligatoire, pas de parallélisme GPU pur
- **Memory leaks** : Surveillance continue requise entre workers
- **Conflits Git** : Résolution manuelle si branches divergent
- **Ressources système** : 64GB RAM + 20 threads CPU requis minimum
- **Tests système** : Phase 4 reste séquentielle (validation intégrité)

---

## 🔧 SPÉCIFICATIONS TECHNIQUES

### Configuration GPU Standard Obligatoire
```python
# STANDARD OBLIGATOIRE - À INTÉGRER DANS CHAQUE SCRIPT
import os
import torch

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB sur Bus PCI 1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Force l'ordre du bus physique
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:0 après mapping)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"🔒 CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER')}")
```

### Validation Obligatoire - AUCUNE EXCEPTION
```python
def validate_rtx3090_mandatory():
    """Validation systématique - OBLIGATOIRE dans chaque script"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    # CONTRÔLE 1: Variables environnement
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    
    # CONTRÔLE 2: GPU physique détecté (après mapping, cuda:0 = RTX 3090)
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in gpu_name:
        raise RuntimeError(f"🚫 GPU détecté: {gpu_name} - RTX 3090 requise")
    
    # CONTRÔLE 3: Mémoire GPU
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 ≈ 24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) insuffisante - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory:.1f}GB)")

# APPELER OBLIGATOIREMENT dans __main__ ou au début du script
if __name__ == "__main__":
    validate_rtx3090_mandatory()
```

### Fichiers à Corriger (40 identifiés - Liste Exhaustive)

#### Modules Core Critiques Initiaux (7)
- `benchmarks/benchmark_stt_realistic.py`
- `LLM/llm_manager_enhanced.py`  
- `LUXA_TTS/tts_handler_coqui.py`
- `Orchestrator/fallback_manager.py`
- `STT/vad_manager_optimized.py`
- `TTS/tts_handler_coqui.py`
- `TTS/tts_handler_piper_native.py`

#### Modules Core Supplémentaires (6)
- `STT/stt_manager_robust.py`
- `STT/vad_manager.py`
- `TTS/tts_handler_piper_espeak.py`
- `TTS/tts_handler_piper_fixed.py`
- `TTS/tts_handler_piper_french.py`
- `utils/gpu_manager.py`

#### Scripts de Test Initiaux (13)
- `tests/test_double_check_corrections.py`
- `tests/test_double_check_validation_simple.py`
- `test_cuda_debug.py`
- `test_cuda.py`
- `test_espeak_french.py`
- `test_french_voice.py`
- `test_gpu_correct.py`
- `test_piper_native.py`
- `test_tts_fixed.py`
- `test_tts_long_feedback.py`
- `test_upmc_model.py`
- `test_validation_decouverte.py`
- `TTS/tts_handler_piper_rtx3090.py`

#### Tests Supplémentaires (2)
- `tests/test_llm_handler.py`
- `tests/test_stt_handler.py`

#### Scripts de Validation Exhaustifs (12)
- `test_correction_validation_1.py`
- `test_correction_validation_2.py`
- `test_correction_validation_3.py`
- `test_correction_validation_4.py`
- `test_rtx3090_detection.py`
- `test_tts_rtx3090_performance.py`
- `test_validation_globale_finale.py`
- `test_validation_mvp_settings.py`
- `test_validation_rtx3090_detection.py`
- `test_validation_stt_manager_robust.py`
- `test_validation_tts_performance.py`
- `validate_gpu_config.py`

---

## 🛠️ MÉTHODOLOGIE DE CORRECTION

### Phase 1 : Préparation et Analyse
1. **Analyse détaillée** de chaque fichier cible
2. **Vérification de la configuration existante** (CUDA_VISIBLE_DEVICES et CUDA_DEVICE_ORDER)
3. **Création de tests de référence** (version originale)
4. **Documentation des fonctionnalités** existantes complètes
5. **Sauvegarde versions originales** dans Git

### Phase 2 : Correction Systématique
1. **Ajout/complétion de la configuration GPU** :
   - `CUDA_VISIBLE_DEVICES='1'`
   - `CUDA_DEVICE_ORDER='PCI_BUS_ID'`
2. **Vérification que le code utilise `cuda:0`** (mappé vers RTX 3090)
3. **Ajout de la validation obligatoire** RTX 3090
4. **Préservation intégrale** de toute la logique métier
5. **Application du template standard** GPU

### Phase 3 : Validation Intégrale - ZÉRO RÉGRESSION
Pour **CHAQUE fichier corrigé** :

#### Test 1 : Configuration GPU (OBLIGATOIRE)
```python
def test_gpu_configuration():
    # ÉTAPE 0: Script diagnostic OBLIGATOIRE POUR CHAQUE FICHIER
    import subprocess
    print("🔍 DIAGNOSTIC RTX 3090 POUR CE FICHIER:")
    result = subprocess.run([
        "python", "C:\\Dev\\SuperWhisper_V6\\test_diagnostic_rtx3090.py"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    assert result.returncode == 0, "ÉCHEC: Script diagnostic RTX 3090"
    assert "RTX 3090 détecté: ✅ OUI" in result.stdout, "ÉCHEC: RTX 3090 non détectée"
    print("✅ Script diagnostic RTX 3090 validé pour ce fichier")
    
    # ÉTAPE 1: Vérifications environnement
    # Vérifier CUDA_VISIBLE_DEVICES='1'
    # Vérifier CUDA_DEVICE_ORDER='PCI_BUS_ID'
    # Vérifier torch.cuda.get_device_name(0) contient "RTX 3090"
    # Vérifier mémoire GPU > 20GB
    # AUCUNE ASSOMPTION - CONTRÔLE FACTUEL OBLIGATOIRE
```

#### Test 2 : Fonctionnalité Intégrale (OBLIGATOIRE)
```python
def test_all_functionalities():
    # Tester 100% des fonctions du module
    # Tester 100% des classes du module  
    # Tester tous les workflows d'usage
    # Comparer sorties avec version originale
    # Vérifier performance identique ou meilleure
    # Valider gestion d'erreurs identique
```

#### Test 3 : Non-Régression (OBLIGATOIRE)
```python
def test_no_regression():
    # Benchmark performance vs version originale
    # Test mémoire (pas de fuites)
    # Test stabilité sur durée
    # Validation sorties bit-perfect si possible
```

### Phase 4 : Documentation et Standards
1. **Rapport de correction détaillé** par fichier
2. **Standards GPU définitifs** pour développements futurs
3. **Guide de validation GPU** 
4. **Template de code** pour nouveaux développements

---

## 🔍 CRITÈRES D'ACCEPTATION STRICTS

### ✅ Correction Validée UNIQUEMENT Si
1. **Configuration GPU complète** : CUDA_VISIBLE_DEVICES='1' ET CUDA_DEVICE_ORDER='PCI_BUS_ID'
2. **RTX 3090 détectée et utilisée** : Validation factuelle obligatoire
3. **Code utilise `cuda:0`** de manière cohérente (mappé vers RTX 3090)
4. **100% fonctionnalités opérationnelles** : aucune régression autorisée
5. **Performance maintenue** : identique ou améliorée (±2% max)
6. **Tests automatisés** : 100% passent
7. **Documentation** : complète et validée

### ❌ Correction Rejetée Si
- **Configuration incomplète** (manque CUDA_DEVICE_ORDER)
- **Mauvaise variable CUDA_VISIBLE_DEVICES** (différente de '1')
- **Une seule fonction** défaillante
- **Régression de performance** détectée (>2%)
- **Validation GPU** échoue
- **Tests automatisés** en échec
- **Comportement modifié** vs original

---

## 🧰 OUTILS ET TECHNOLOGIES REQUIS

### Langages
- **Python 3.8+** pour scripts de correction
- **PowerShell 7+** pour automation Windows

### Bibliothèques Python
- **PyTorch** pour validation GPU
- **pathlib** pour gestion fichiers
- **unittest/pytest** pour tests automatisés
- **psutil** pour monitoring système
- **memory_profiler** pour validation mémoire

### Outils de Développement
- **Git** pour versioning et rollback sécurisé
- **Cursor/VS Code** pour édition
- **TaskMaster** pour gestion des tâches structurées

### Validation et Tests
- **Scripts de test personnalisés** par fichier
- **Profiling mémoire GPU** (nvidia-smi)
- **Benchmarks de performance** comparatifs
- **Tests de charge** pour validation stabilité

---

## ⚠️ RISQUES ET MITIGATION

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Configuration GPU incomplète | **CRITIQUE** | Élevé | Double validation (CUDA_VISIBLE_DEVICES + CUDA_DEVICE_ORDER) |
| Régression fonctionnelle | **CRITIQUE** | Moyen | Tests exhaustifs avant/après + rollback Git |
| Performance dégradée | **ÉLEVÉ** | Faible | Benchmarks comparatifs + validation continue |
| Erreurs silencieuses | **ÉLEVÉ** | Moyen | Validation GPU obligatoire + tests automatisés |
| Mauvaise compréhension config | **ÉLEVÉ** | Élevé | Documentation claire + exemples concrets |

---

## 📈 MÉTRIQUES DE SUCCÈS

### Objectifs Quantifiables
- **100%** des 40 fichiers avec configuration GPU complète
- **100%** des fichiers utilisent RTX 3090 exclusivement
- **0** régression fonctionnelle détectée
- **100%** des tests automatisés passent
- **≥98%** de préservation des performances
- **Standards GPU** documentés et validés

### Livrables Attendus
1. **40 fichiers avec configuration GPU homogène**
2. **Code utilisant `cuda:0` de manière cohérente**
3. **40 validations script diagnostic** "C:\Dev\SuperWhisper_V6\test_diagnostic_rtx3090.py"
4. **Standards GPU documentés** 
5. **Guide de développement** GPU
6. **Tests automatisés** pour validation continue
7. **Rapport de correction** détaillé

---

## 🎯 DÉFINITION DU SUCCÈS

**Le projet sera considéré comme réussi quand :**
- Tous les scripts ont la configuration GPU complète (CUDA_VISIBLE_DEVICES='1' + CUDA_DEVICE_ORDER='PCI_BUS_ID')
- Tous les scripts utilisent exclusivement la RTX 3090
- Le code utilise `cuda:0` de manière cohérente
- Aucune régression fonctionnelle n'est détectée
- Les standards GPU sont adoptés pour les développements futurs
- La validation GPU est automatisée et obligatoire
- La documentation est complète et accessible

---

**Ce PRD garantit une approche méthodique, rigoureuse et sans risque pour l'homogénisation du mapping GPU, avec validation factuelle obligatoire à chaque étape et préservation intégrale des fonctionnalités existantes.** 