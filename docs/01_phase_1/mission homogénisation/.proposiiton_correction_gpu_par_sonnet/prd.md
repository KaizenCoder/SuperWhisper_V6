# 📋 PRD - HOMOGÉNISATION DU MAPPING GPU SUPERWHISPER V6

---

**Projet :** Correction et Homogénisation du Mapping GPU SuperWhisper V6  
**Version :** 1.0  
**Date :** Décembre 2024  
**Priorité :** CRITIQUE  
**Durée estimée :** 15 heures  

---

## 🎯 CONTEXTE ET PROBLÉMATIQUE

### Situation Actuelle
Le projet SuperWhisper V6 présente une **méthodologie de sélection et contrôle GPU non homogène** à travers ses 89 fichiers Python/PowerShell. Cette hétérogénéité génère :

- **Risques de performance** : Utilisation accidentelle de RTX 5060 Ti (16GB) au lieu de RTX 3090 (24GB)
- **Instabilité système** : Mappings GPU incohérents entre modules
- **Maintenance complexe** : Pas de standard unifié para la sélection GPU
- **Erreurs silencieuses** : Aucune validation systématique du GPU utilisé

### Découverte Critique
L'analyse factuelle révèle **20 scripts utilisant un mapping GPU erroné** :
- **Erreur** : `CUDA_VISIBLE_DEVICES='1'` (RTX 5060 Ti)  
- **Correct** : `CUDA_VISIBLE_DEVICES='0'` (RTX 3090)

### Configuration Matérielle
```
RTX 3090 (24GB) → CUDA:0 → SEULE GPU AUTORISÉE
RTX 5060 Ti (16GB) → CUDA:1 → STRICTEMENT INTERDITE
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
1. **Corriger les 20 scripts identifiés** avec mapping GPU erroné
2. **Implémenter une validation systématique** de sélection GPU
3. **Standardiser la méthodologie** de contrôle GPU
4. **Garantir zéro régression fonctionnelle**
5. **Documenter les standards** pour développements futurs

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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:0) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
```

### Validation Obligatoire - AUCUNE EXCEPTION
```python
def validate_rtx3090_mandatory():
    """Validation systématique - OBLIGATOIRE dans chaque script"""
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    # CONTRÔLE 1: Variable environnement
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '0':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '0'")
    
    # CONTRÔLE 2: GPU physique détecté  
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

### Fichiers à Corriger (20 identifiés)

#### Modules Core Critiques (7)
- `benchmarks/benchmark_stt_realistic.py`
- `LLM/llm_manager_enhanced.py`  
- `LUXA_TTS/tts_handler_coqui.py`
- `Orchestrator/fallback_manager.py`
- `STT/vad_manager_optimized.py`
- `TTS/tts_handler_coqui.py`
- `TTS/tts_handler_piper_native.py`

#### Scripts de Test (13)
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

---

## 🛠️ MÉTHODOLOGIE DE CORRECTION

### Phase 1 : Préparation et Analyse
1. **Analyse détaillée** de chaque fichier cible
2. **Création de tests de référence** (version originale)
3. **Documentation des fonctionnalités** existantes complètes
4. **Sauvegarde versions originales** dans Git

### Phase 2 : Correction Systématique
1. **Modification du mapping GPU** (`cuda:1` → `cuda:0`)
2. **Ajout de la validation obligatoire** RTX 3090
3. **Préservation intégrale** de toute la logique métier
4. **Application du template standard** GPU

### Phase 3 : Validation Intégrale - ZÉRO RÉGRESSION
Pour **CHAQUE fichier corrigé** :

#### Test 1 : Configuration GPU (OBLIGATOIRE)
```python
def test_gpu_configuration():
    # Vérifier CUDA_VISIBLE_DEVICES='0'
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
1. **Configuration GPU vérifiée factuellement** : RTX 3090 détectée et utilisée
2. **100% fonctionnalités opérationnelles** : aucune régression autorisée
3. **Performance maintenue** : identique ou améliorée (±2% max)
4. **Tests automatisés** : 100% passent
5. **Validation comparative** : comportement identique à l'original
6. **Documentation** : complète et validée

### ❌ Correction Rejetée Si
- **Une seule fonction** défaillante
- **Régression de performance** détectée (>2%)
- **Validation GPU** échoue
- **Tests automatisés** en échec
- **Comportement modifié** vs original
- **Fuite mémoire** détectée

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
| Régression fonctionnelle | **CRITIQUE** | Moyen | Tests exhaustifs avant/après + rollback Git |
| Performance dégradée | **ÉLEVÉ** | Faible | Benchmarks comparatifs + validation continue |
| Erreurs silencieuses | **ÉLEVÉ** | Moyen | Validation GPU obligatoire + tests automatisés |
| Rollback complexe | **MOYEN** | Faible | Versioning Git + sauvegarde branches |
| Instabilité système | **CRITIQUE** | Faible | Tests de charge + validation mémoire |

---

## 📈 MÉTRIQUES DE SUCCÈS

### Objectifs Quantifiables
- **100%** des 20 fichiers corrigés utilisent RTX 3090
- **0** régression fonctionnelle détectée
- **100%** des tests automatisés passent
- **≥98%** de préservation des performances
- **Standards GPU** documentés et validés

### Livrables Attendus
1. **20 fichiers corrigés** et validés
2. **Standards GPU documentés** 
3. **Guide de développement** GPU
4. **Tests automatisés** pour validation continue
5. **Rapport de correction** détaillé

---

## 🎯 DÉFINITION DU SUCCÈS

**Le projet sera considéré comme réussi quand :**
- Tous les scripts utilisent exclusivement la RTX 3090
- Aucune régression fonctionnelle n'est détectée
- Les standards GPU sont adoptés pour les développements futurs
- La validation GPU est automatisée et obligatoire
- La documentation est complète et accessible

---

**Ce PRD garantit une approche méthodique, rigoureuse et sans risque pour l'homogénisation du mapping GPU, avec validation factuelle obligatoire à chaque étape et préservation intégrale des fonctionnalités existantes.** 