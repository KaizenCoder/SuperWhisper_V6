# 🎯 PROMPT MAÎTRE - HOMOGÉNISATION GPU SUPERWHISPER V6

**Mission :** Corriger la méthodologie de sélection et contrôle GPU non homogène dans SuperWhisper V6  
**Criticité :** MAXIMALE - Impact direct sur performance et stabilité système  
**Résultat attendu :** 40 fichiers corrigés avec validation factuelle intégrale et zéro régression  

---

## 🎪 CONTEXTE CRITIQUE DE LA MISSION

### Problématique Identifiée
Le projet SuperWhisper V6 présente une **méthodologie de sélection et contrôle GPU non homogène** qui génère :
- **Risques de performance** : Utilisation accidentelle RTX 5060 Ti (16GB) au lieu de RTX 3090 (24GB)
- **Instabilité système** : Mappings GPU incohérents entre modules
- **Erreurs silencieuses** : Absence de validation systématique du GPU utilisé

### Configuration Matérielle CRITIQUE
```
🎮 Configuration physique du système :
- GPU Bus PCI 0 : RTX 5060 Ti (16GB) ❌ STRICTEMENT INTERDITE
- GPU Bus PCI 1 : RTX 3090 (24GB) ✅ SEULE GPU AUTORISÉE

⚠️ ATTENTION : PyTorch ordonne les GPU différemment sans CUDA_DEVICE_ORDER='PCI_BUS_ID'
```

### Découverte Factuelle
**40 scripts identifiés** nécessitent une homogénéisation :
- **Configuration requise** : 
  ```python
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Sélectionne RTX 3090 sur bus PCI 1
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Force l'ordre physique
  ```
- **Après cette configuration** : `cuda:0` dans le code = RTX 3090 (remapping PyTorch)

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

## 🎯 OBJECTIF DE LA MISSION

**Homogénéiser et sécuriser la sélection GPU** dans tous les scripts identifiés pour garantir l'utilisation exclusive de la RTX 3090, avec :
1. **Ajout de la configuration environnement complète** dans chaque fichier
2. **Utilisation cohérente de `cuda:0`** dans le code (qui pointera vers RTX 3090)
3. **Validation factuelle obligatoire** pour chaque correction
4. **Préservation intégrale** des fonctionnalités (zéro régression)
5. **Standardisation de la méthodologie** pour développements futurs

---

## 🚀 NOUVELLES OPTIMISATIONS VALIDÉES

### Memory Leak Solution V4.0 - OBLIGATOIRE
- **Script central** : `memory_leak_v4.py` (solution finalisée)
- **Cleanup automatique** pour tous tests GPU avec context manager
- **Monitoring temps réel** : mémoire, fragmentation, performance
- **Queue GPU exclusive** pour parallélisation sécurisée
- **Métriques Prometheus** intégrées pour monitoring

### Parallélisation Validée - 64% GAIN PERFORMANCE
- **Configuration système validée** : 64GB RAM + 20 CPU threads + RTX 3090
- **Gain performance confirmé** : 33h → 12-16h (64% plus rapide)
- **Architecture éprouvée** : ThreadPool + GPU Queue + Memory Management automatique
- **Tests validés** : 10/10 stress tests réussis, 0% memory leak détecté

### Integration Workflow Memory Leak Prevention
1. **Utiliser `@gpu_test_cleanup()`** pour TOUS les tests GPU
2. **Queue GPU exclusive** avec sémaphore multiprocess
3. **Memory monitoring automatique** avec seuils paramétrables
4. **Fallback séquentiel** en cas d'instabilité parallélisation
5. **Logs JSON** avec rollover automatique pour audit
6. **Emergency reset** automatique si memory leak critique

### Configuration Memory Management
```python
# Import obligatoire pour tous fichiers avec GPU
from memory_leak_v4 import gpu_test_cleanup, validate_no_memory_leak

# Décorateur obligatoire pour TOUS tests GPU
@gpu_test_cleanup("nom_test_descriptif")
def your_gpu_test_function():
    device = "cuda:0"  # RTX 3090 après mapping
    # Votre code GPU ici
    # Cleanup automatique à la fin du context

# Validation obligatoire en fin de script
if __name__ == "__main__":
    validate_rtx3090_mandatory()  # Validation GPU
    # Tests...
    validate_no_memory_leak()     # Validation memory leak
```

### Parallélisation Architecture
- **Phase 2 + 3** : 13 modules core + 27 scripts test = **PARALLÉLISABLES**
- **Queue GPU exclusive** : Un seul script accède GPU à la fois
- **Memory cleanup** : Automatique entre chaque script
- **Git branches** : Dédiées par thread pour éviter conflits
- **Monitoring centralisé** : Métriques temps réel via Prometheus
- **Recovery automatique** : Emergency reset si memory leak détecté

---

## 🛠️ OUTILS ET TECHNOLOGIES À UTILISER

### Outils de Développement
- **Git** : Versioning et rollback sécurisé (branche dédiée obligatoire)
- **Python 3.8+** : Langage principal pour corrections et validations
- **PyTorch** : Validation GPU et détection matériel
- **PowerShell 7+** : Automation scripts Windows
- **Cursor/VS Code** : Édition de code

### Bibliothèques Python Requises
```python
import os
import torch
import time
import psutil
from pathlib import Path
import unittest
```

### Outils de Validation
- **Scripts de test personnalisés** par fichier
- **nvidia-smi** : Monitoring GPU
- **memory_profiler** : Validation mémoire
- **Benchmarks comparatifs** performance

---

## 📋 LISTE EXHAUSTIVE DES FICHIERS À CORRIGER (40)

### Modules Core Critiques Initiaux (7)
```
📁 benchmarks/benchmark_stt_realistic.py
📁 LLM/llm_manager_enhanced.py
📁 LUXA_TTS/tts_handler_coqui.py
📁 Orchestrator/fallback_manager.py
📁 STT/vad_manager_optimized.py
📁 TTS/tts_handler_coqui.py
📁 TTS/tts_handler_piper_native.py
```

### Modules Core Supplémentaires (6)
```
📁 STT/stt_manager_robust.py
📁 STT/vad_manager.py
📁 TTS/tts_handler_piper_espeak.py
📁 TTS/tts_handler_piper_fixed.py
📁 TTS/tts_handler_piper_french.py
📁 utils/gpu_manager.py
```

### Scripts de Test Initiaux (13)
```
📁 tests/test_double_check_corrections.py
📁 tests/test_double_check_validation_simple.py
📁 test_cuda_debug.py
📁 test_cuda.py
📁 test_espeak_french.py
📁 test_french_voice.py
📁 test_gpu_correct.py
📁 test_piper_native.py
📁 test_tts_fixed.py
📁 test_tts_long_feedback.py
📁 test_upmc_model.py
📁 test_validation_decouverte.py
📁 TTS/tts_handler_piper_rtx3090.py
```

### Tests Supplémentaires (2)
```
📁 tests/test_llm_handler.py
📁 tests/test_stt_handler.py
```

### Scripts de Validation Exhaustifs (12)
```
📁 test_correction_validation_1.py
📁 test_correction_validation_2.py
📁 test_correction_validation_3.py
📁 test_correction_validation_4.py
📁 test_rtx3090_detection.py
📁 test_tts_rtx3090_performance.py
📁 test_validation_globale_finale.py
📁 test_validation_mvp_settings.py
📁 test_validation_rtx3090_detection.py
📁 test_validation_stt_manager_robust.py
📁 test_validation_tts_performance.py
📁 validate_gpu_config.py
```

---

## 🔧 TEMPLATE DE CORRECTION OBLIGATOIRE V2.0 [avec Memory Leak V4.0]

### Configuration GPU + Memory Management - À INTÉGRER DANS CHAQUE SCRIPT
```python
#!/usr/bin/env python3
"""
[Description du script]
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:0) + Memory Leak Prevention V4.0 OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 3090 (Bus PCI 1) = SEULE AUTORISÉE - RTX 5060 Ti (Bus PCI 0) = INTERDITE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB sur Bus PCI 1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Force l'ordre du bus physique
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:0 après mapping)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"🔒 CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER')}")

# =============================================================================
# 🚨 MEMORY LEAK PREVENTION V4.0 - OBLIGATOIRE 
# =============================================================================
# Import du système de prévention memory leak validé
try:
    from memory_leak_v4 import (
        configure_for_environment, 
        gpu_test_cleanup, 
        validate_no_memory_leak,
        emergency_gpu_reset
    )
    # Configuration environnement (dev/ci/production)
    configure_for_environment("dev")  # Adapter selon contexte
    print("✅ Memory Leak Prevention V4.0 activé")
except ImportError:
    print("⚠️ Memory Leak V4.0 non disponible - Continuer avec validation standard")
    gpu_test_cleanup = lambda name: lambda func: func  # Fallback

# Maintenant imports normaux...
import torch
# ... autres imports
```

### Fonction de Validation OBLIGATOIRE
```python
def validate_rtx3090_mandatory():
    """Validation systématique RTX 3090 - OBLIGATOIRE dans chaque script"""
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

---

## 🔬 MÉTHODOLOGIE DE CORRECTION - ENSEIGNEMENTS CRITIQUES

### LEÇON MAÎTRESSE : VALIDATION FACTUELLE OBLIGATOIRE
**❌ ERREUR À ÉVITER :** Assumer que la configuration est correcte  
**✅ MÉTHODE CORRECTE :** Contrôler factuellement à chaque étape

### Processus de Correction par Fichier (50min/fichier core, 30min/test)

#### Étape 1 : Préparation (10min)
```bash
# Sauvegarder version originale
cp [fichier] docs/gpu-correction/backups/[fichier].backup

# Analyser le fichier
grep -n "cuda" [fichier]
grep -n "CUDA_VISIBLE_DEVICES" [fichier]
grep -n "CUDA_DEVICE_ORDER" [fichier]
grep -n "device" [fichier]
```

#### Étape 2 : Correction Configuration GPU (15min)
```python
# AJOUTS/MODIFICATIONS STANDARDS
# 1. Ajouter en début de fichier (après shebang et docstring) :
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# 2. Dans le code, utiliser :
device = "cuda:0"  # ou "cuda" (équivalent après mapping)
torch.device("cuda:0")
torch.cuda.set_device(0)
device_map={"": 0}
gpu_device_index: 0
```

#### Étape 3 : Validation Factuelle OBLIGATOIRE (15min)
```python
def test_fichier_correction():
    """Test factuel - AUCUNE ASSOMPTION AUTORISÉE"""
    
    # TEST 0: Script diagnostic OBLIGATOIRE POUR CHAQUE FICHIER
    import subprocess
    print("🔍 DIAGNOSTIC RTX 3090 POUR CE FICHIER:")
    result = subprocess.run([
        "python", "C:\\Dev\\SuperWhisper_V6\\test_diagnostic_rtx3090.py"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    assert result.returncode == 0, "ÉCHEC: Script diagnostic RTX 3090"
    assert "RTX 3090 détecté: ✅ OUI" in result.stdout, "ÉCHEC: RTX 3090 non détectée"
    print("✅ Script diagnostic RTX 3090 validé pour ce fichier")
    
    # TEST 1: Configuration environnement
    cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES')
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER')
    assert cuda_env == '1', f"ÉCHEC: CUDA_VISIBLE_DEVICES='{cuda_env}' au lieu de '1'"
    assert cuda_order == 'PCI_BUS_ID', f"ÉCHEC: CUDA_DEVICE_ORDER='{cuda_order}' au lieu de 'PCI_BUS_ID'"
    
    # TEST 2: GPU physique détecté
    gpu_name = torch.cuda.get_device_name(0)
    assert "RTX 3090" in gpu_name, f"ÉCHEC: GPU détecté '{gpu_name}' au lieu de RTX 3090"
    
    # TEST 3: Mémoire GPU
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    assert gpu_memory > 20, f"ÉCHEC: GPU {gpu_memory:.1f}GB au lieu de ~24GB"
    
    print(f"✅ Configuration GPU validée: {gpu_name} ({gpu_memory:.1f}GB)")
```

#### Étape 4 : Test Fonctionnalité Intégrale (10min)
```python
def test_fonctionnalite_integrale():
    """Test TOUTES les fonctionnalités - ZÉRO RÉGRESSION AUTORISÉE"""
    
    # Importer le module corrigé
    import [module_corrigé]
    
    # Tester toutes les classes
    for classe in [module_corrigé].get_classes():
        instance = classe()
        # Validation constructeur, méthodes, propriétés
    
    # Tester toutes les fonctions
    for fonction in [module_corrigé].get_functions():
        # Validation avec paramètres réels
        # Comparaison sorties avec version originale
    
    # Test performance (pas de régression > 2%)
    performance_actuelle = benchmark_fonction()
    assert performance_actuelle >= performance_reference * 0.98
    
    print("✅ Toutes fonctionnalités validées - aucune régression")
```

---

## ⚠️ CRITÈRES D'ACCEPTATION STRICTS

### ✅ Correction VALIDÉE UNIQUEMENT Si
1. **Configuration GPU complète** : CUDA_VISIBLE_DEVICES='1' ET CUDA_DEVICE_ORDER='PCI_BUS_ID'
2. **RTX 3090 détectée et utilisée** : Validation factuelle obligatoire
3. **100% des fonctionnalités opérationnelles** : aucune régression autorisée
4. **Performance maintenue** : identique ou améliorée (±2% maximum)
5. **Tests automatisés** : 100% passent
6. **Documentation** : rapport de correction complet

### ❌ Correction REJETÉE Si
- **Configuration incomplète** (manque CUDA_DEVICE_ORDER)
- **Une seule fonction** défaillante
- **Régression de performance** détectée (>2%)
- **Validation GPU** échoue
- **Tests automatisés** en échec
- **Comportement modifié** vs original

---

## 🔍 TEMPLATES DE VALIDATION PAR TYPE DE FICHIER

### Pour Modules STT/VAD
```python
def test_stt_vad_module():
    # DIAGNOSTIC RTX 3090 OBLIGATOIRE
    import subprocess
    result = subprocess.run([
        "python", "C:\\Dev\\SuperWhisper_V6\\test_diagnostic_rtx3090.py"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Script diagnostic RTX 3090 échoué"
    print("✅ Diagnostic RTX 3090 validé pour module STT/VAD")
    
    validate_rtx3090_mandatory()
    
    # Test spécifique STT/VAD
    vad = VADManager()
    result = vad.detect_speech(test_audio)
    assert result is not None
    assert result.confidence > 0.8
    
    # Test performance
    start = time.time()
    vad.process_batch(audio_batch)
    duration = time.time() - start
    assert duration <= reference_duration * 1.02
```

### Pour Modules TTS
```python
def test_tts_module():
    # DIAGNOSTIC RTX 3090 OBLIGATOIRE
    import subprocess
    result = subprocess.run([
        "python", "C:\\Dev\\SuperWhisper_V6\\test_diagnostic_rtx3090.py"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Script diagnostic RTX 3090 échoué"
    print("✅ Diagnostic RTX 3090 validé pour module TTS")
    
    validate_rtx3090_mandatory()
    
    # Test spécifique TTS
    tts = TTSHandler()
    audio = tts.synthesize("Test phrase française")
    assert len(audio) > 0
    assert audio_quality(audio) > min_quality_threshold
    
    # Test voix multiples
    for voice in available_voices:
        tts.set_voice(voice)
        audio = tts.synthesize("Test")
        assert audio_voice_correct(audio, voice)
```

### Pour Modules LLM
```python
def test_llm_module():
    # DIAGNOSTIC RTX 3090 OBLIGATOIRE
    import subprocess
    result = subprocess.run([
        "python", "C:\\Dev\\SuperWhisper_V6\\test_diagnostic_rtx3090.py"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Script diagnostic RTX 3090 échoué"
    print("✅ Diagnostic RTX 3090 validé pour module LLM")
    
    validate_rtx3090_mandatory()
    
    # Test spécifique LLM
    llm = LLMManager()
    response = llm.process("Question test")
    assert response is not None
    assert len(response) > 0
    assert response_quality(response) > min_threshold
```

### Pour Scripts de Test
```python
def test_script_test():
    # DIAGNOSTIC RTX 3090 OBLIGATOIRE
    import subprocess
    result = subprocess.run([
        "python", "C:\\Dev\\SuperWhisper_V6\\test_diagnostic_rtx3090.py"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Script diagnostic RTX 3090 échoué"
    print("✅ Diagnostic RTX 3090 validé pour script de test")
    
    validate_rtx3090_mandatory()
    
    # Exécuter le script de test
    result = run_test_script()
    assert result.exit_code == 0
    assert result.gpu_used == "RTX 3090"
    assert no_errors_in_output(result.output)
```

---

## 📊 WORKFLOW DE CORRECTION RECOMMANDÉ

### Phase 1 : Setup Sécurisé
```bash
# Créer branche dédiée
git checkout -b feature/gpu-mapping-homogenization
git push -u origin feature/gpu-mapping-homogenization

# Créer structure de travail
mkdir -p docs/gpu-correction/{reports,tests,backups}

# Tag de référence
git tag -a v-before-gpu-correction -m "État avant correction mapping GPU"
```

### Phase 2 : Correction Systématique
1. **Vérifier la configuration existante** dans chaque fichier
2. **Ajouter/compléter la configuration GPU** si nécessaire
3. **S'assurer que le code utilise `cuda:0`** après la configuration
4. **Validation factuelle** à chaque étape
5. **Documentation** des résultats

### Phase 3 : Validation Système
```python
def test_systeme_complet():
    """Test d'intégration finale"""
    
    # Vérifier tous les processus GPU
    for process in get_gpu_processes():
        assert process.gpu_id == 0  # cuda:0 après mapping
        assert "RTX 3090" in process.gpu_name
    
    # Test workflow complet STT→LLM→TTS
    result = run_complete_workflow(test_audio)
    assert result.success
    assert result.all_gpu_rtx3090
    
    print("✅ Système complet validé RTX 3090")
```

---

## 📈 RAPPORT DE VALIDATION OBLIGATOIRE

### Template de Rapport par Fichier
```markdown
## RAPPORT - [nom_fichier]

### Configuration GPU
- ✅ CUDA_VISIBLE_DEVICES: '1'
- ✅ CUDA_DEVICE_ORDER: 'PCI_BUS_ID'
- ✅ GPU détecté: NVIDIA GeForce RTX 3090 (24.0GB)
- ✅ Validation fonction: validate_rtx3090_mandatory() 

### Tests Fonctionnels
- ✅ Import module: OK
- ✅ Classes testées: X/X (100%)
- ✅ Fonctions testées: Y/Y (100%)
- ✅ Workflows testées: Z/Z (100%)

### Performance
- ✅ Temps exécution: [durée] (ref: [référence])
- ✅ Mémoire GPU: [usage] (max: [limite])
- ✅ Régression: 0% (amélioration: +X%)

### Validation Comparative
- ✅ Sorties identiques: OUI
- ✅ Comportement identique: OUI
- ✅ Gestion erreurs: IDENTIQUE

### Conclusion
✅ CORRECTION VALIDÉE - Aucune régression détectée
```

---

## 🚨 POINTS D'ATTENTION CRITIQUES

### Configuration GPU Correcte
```python
# ✅ CORRECT - Force RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# Après cette config : cuda:0 = RTX 3090

# ❌ INCORRECT - Manque CUDA_DEVICE_ORDER
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Sans CUDA_DEVICE_ORDER, l'ordre GPU peut être imprévisible
```

### Erreurs à Éviter Absolument
1. **❌ Oublier `CUDA_DEVICE_ORDER='PCI_BUS_ID'`** - Critical !
2. **❌ Assumer que la correction fonctionne** sans test factuel
3. **❌ Tester seulement la "fonctionnalité principale"** au lieu de TOUT
4. **❌ Ignorer les tests de performance** comparatifs
5. **❌ Ne pas documenter** les résultats de validation

### Validation Factuelle Obligatoire
```python
# TOUJOURS VÉRIFIER FACTUELLEMENT
gpu_name = torch.cuda.get_device_name(0)
assert "RTX 3090" in gpu_name, f"ÉCHEC: {gpu_name}"

# JAMAIS D'ASSOMPTION
# ❌ INTERDIT: "La configuration devrait être correcte"
# ✅ OBLIGATOIRE: Contrôle factuel à chaque étape
```

### Gestion des Erreurs
```python
try:
    validate_rtx3090_mandatory()
    run_functionality_tests()
    check_performance_regression()
except Exception as e:
    # Rollback automatique
    restore_original_file()
    raise RuntimeError(f"Correction échouée: {e}")
```

---

## 🎯 LIVRABLES ATTENDUS

1. **40 fichiers avec configuration GPU homogène** (CUDA_VISIBLE_DEVICES='1' + CUDA_DEVICE_ORDER='PCI_BUS_ID')
2. **Code utilisant `cuda:0`** de manière cohérente
3. **40 validations script diagnostic** "C:\Dev\SuperWhisper_V6\test_diagnostic_rtx3090.py"
4. **40 rapports de validation** détaillés
5. **Tests automatisés** pour chaque correction
6. **Standards GPU documentés** pour développements futurs
7. **Guide de validation** réutilisable
8. **Rapport final** de mission

---

## 🚀 COMMANDES D'EXÉCUTION

### Lancement de la Mission
```bash
# Exécuter avec TaskMaster
task-master parse-prd --input=docs/prd.md
task-master analyze-complexity --research
task-master list
task-master next

# Script de validation GPU
python scripts/validate_gpu_configuration.py

# Ou exécution directe
python -c "exec(open('docs/prompt.md').read())"
```

---

**CE PROMPT GARANTIT UNE EXÉCUTION MÉTHODIQUE, RIGOUREUSE ET SANS RISQUE DE LA MISSION D'HOMOGÉNISATION GPU, AVEC VALIDATION FACTUELLE OBLIGATOIRE À CHAQUE ÉTAPE ET PRÉSERVATION INTÉGRALE DES FONCTIONNALITÉS EXISTANTES.**

---

*Prompt créé d'après les enseignements méthodologiques tirés de la session de travail SuperWhisper V6 - Mise à jour Juin 2025* 