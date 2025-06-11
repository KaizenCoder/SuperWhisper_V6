# 🎯 PROMPT MAÎTRE - HOMOGÉNISATION GPU SUPERWHISPER V6

**Mission :** Corriger la méthodologie de sélection et contrôle GPU non homogène dans SuperWhisper V6  
**Criticité :** MAXIMALE - Impact direct sur performance et stabilité système  
**Résultat attendu :** 20 fichiers corrigés avec validation factuelle intégrale et zéro régression  

---

## 🎪 CONTEXTE CRITIQUE DE LA MISSION

### Problématique Identifiée
Le projet SuperWhisper V6 présente une **méthodologie de sélection et contrôle GPU non homogène** qui génère :
- **Risques de performance** : Utilisation accidentelle RTX 5060 Ti (16GB) au lieu de RTX 3090 (24GB)
- **Instabilité système** : Mappings GPU incohérents entre modules
- **Erreurs silencieuses** : Absence de validation systématique du GPU utilisé

### Configuration Matérielle CRITIQUE
```
🎮 RTX 3090 (24GB) → CUDA:0 → SEULE GPU AUTORISÉE ✅
🚫 RTX 5060 Ti (16GB) → CUDA:1 → STRICTEMENT INTERDITE ❌
```

### Découverte Factuelle
**20 scripts identifiés** utilisent un mapping GPU erroné :
- **Erreur actuelle** : `CUDA_VISIBLE_DEVICES='1'` (RTX 5060 Ti)
- **Correction requise** : `CUDA_VISIBLE_DEVICES='0'` (RTX 3090)

---

## 🎯 OBJECTIF DE LA MISSION

**Homogénéiser et sécuriser la sélection GPU** dans tous les scripts identifiés pour garantir l'utilisation exclusive de la RTX 3090, avec :
1. **Correction des 20 fichiers** avec mapping GPU erroné
2. **Validation factuelle obligatoire** pour chaque correction
3. **Préservation intégrale** des fonctionnalités (zéro régression)
4. **Standardisation de la méthodologie** pour développements futurs

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

## 📋 LISTE EXHAUSTIVE DES FICHIERS À CORRIGER (20)

### Modules Core Critiques (7)
```
📁 benchmarks/benchmark_stt_realistic.py
📁 LLM/llm_manager_enhanced.py
📁 LUXA_TTS/tts_handler_coqui.py
📁 Orchestrator/fallback_manager.py
📁 STT/vad_manager_optimized.py
📁 TTS/tts_handler_coqui.py
📁 TTS/tts_handler_piper_native.py
```

### Scripts de Test (13)
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

---

## 🔧 TEMPLATE DE CORRECTION OBLIGATOIRE

### Configuration GPU Standard - À INTÉGRER DANS CHAQUE SCRIPT
```python
#!/usr/bin/env python3
"""
[Description du script]
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:0) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 Ti (CUDA:1) = INTERDITE - RTX 3090 (CUDA:0) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:0) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

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
grep -n "device" [fichier]
```

#### Étape 2 : Correction Mapping GPU (15min)
```python
# REMPLACEMENTS STANDARDS
# AVANT (erroné) → APRÈS (correct)
os.environ['CUDA_VISIBLE_DEVICES'] = '1' → os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"cuda:1" → "cuda:0"
torch.device("cuda:1") → torch.device("cuda:0")
torch.cuda.set_device(1) → torch.cuda.set_device(0)
device_map={"": 1} → device_map={"": 0}
gpu_device_index: 1 → gpu_device_index: 0
```

#### Étape 3 : Validation Factuelle OBLIGATOIRE (15min)
```python
def test_fichier_correction():
    """Test factuel - AUCUNE ASSOMPTION AUTORISÉE"""
    
    # TEST 1: Configuration environnement
    cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert cuda_env == '0', f"ÉCHEC: CUDA_VISIBLE_DEVICES='{cuda_env}' au lieu de '0'"
    
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
1. **Configuration GPU vérifiée factuellement** : RTX 3090 détectée et utilisée
2. **100% des fonctionnalités opérationnelles** : aucune régression autorisée
3. **Performance maintenue** : identique ou améliorée (±2% maximum)
4. **Tests automatisés** : 100% passent
5. **Validation comparative** : comportement identique à l'original
6. **Documentation** : rapport de correction complet

### ❌ Correction REJETÉE Si
- **Une seule fonction** défaillante
- **Régression de performance** détectée (>2%)
- **Validation GPU** échoue
- **Tests automatisés** en échec
- **Comportement modifié** vs original
- **Fuite mémoire** détectée

---

## 🔍 TEMPLATES DE VALIDATION PAR TYPE DE FICHIER

### Pour Modules STT/VAD
```python
def test_stt_vad_module():
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
1. **Modules Core en priorité** (7 fichiers)
2. **Scripts de test ensuite** (13 fichiers)
3. **Validation factuelle** à chaque étape
4. **Documentation** des résultats

### Phase 3 : Validation Système
```python
def test_systeme_complet():
    """Test d'intégration finale"""
    
    # Vérifier tous les processus GPU
    for process in get_gpu_processes():
        assert process.gpu_id == 0
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
- ✅ CUDA_VISIBLE_DEVICES: '0'
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

### Erreurs à Éviter Absolument
1. **❌ Assumer que la correction fonctionne** sans test factuel
2. **❌ Tester seulement la "fonctionnalité principale"** au lieu de TOUT
3. **❌ Ignorer les tests de performance** comparatifs
4. **❌ Oublier la validation GPU** dans les scripts modifiés
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

1. **20 fichiers corrigés** avec mapping GPU RTX 3090
2. **20 rapports de validation** détaillés
3. **Tests automatisés** pour chaque correction
4. **Standards GPU documentés** pour développements futurs
5. **Guide de validation** réutilisable
6. **Rapport final** de mission

---

## 🚀 COMMANDES D'EXÉCUTION

### Lancement de la Mission
```bash
# Exécuter avec TaskMaster
task-master parse-prd --input=docs/prd.md
task-master analyze-complexity --research
task-master list
task-master next

# Ou exécution directe
python -c "exec(open('docs/prompt.md').read())"
```

---

**CE PROMPT GARANTIT UNE EXÉCUTION MÉTHODIQUE, RIGOUREUSE ET SANS RISQUE DE LA MISSION D'HOMOGÉNISATION GPU, AVEC VALIDATION FACTUELLE OBLIGATOIRE À CHAQUE ÉTAPE ET PRÉSERVATION INTÉGRALE DES FONCTIONNALITÉS EXISTANTES.**

---

*Prompt créé d'après les enseignements méthodologiques tirés de la session de travail SuperWhisper V6 - Décembre 2024* 