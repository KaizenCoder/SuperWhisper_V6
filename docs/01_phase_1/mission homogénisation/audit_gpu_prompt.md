# 🚨 PROMPT AUDIT CRITIQUE CONFIGURATION GPU - SuperWhisper V6 (LUXA)

## CONTEXTE CRITIQUE
**DÉCOUVERTE MAJEURE** : Configuration GPU massivement incorrecte détectée sur l'ensemble du projet SuperWhisper V6. Audit systématique requis pour identifier et corriger TOUTES les occurrences de mauvaise configuration GPU.

## ⚠️ CONFIGURATION MATÉRIELLE OBLIGATOIRE
- **🚫 RTX 5060 (CUDA:0 / GPU:0)** = **STRICTEMENT INTERDITE** (Port principal, 8GB VRAM)
- **✅ RTX 3090 (CUDA:1 / GPU:1)** = **SEULE AUTORISÉE** (Port secondaire, 24GB VRAM)
- **🔍 SIGNATURE VALIDATION** : 24GB VRAM = RTX 3090, toute autre config = ERREUR CRITIQUE

## MISSION D'AUDIT
Effectuer un contrôle exhaustif de TOUS les fichiers du projet SuperWhisper V6 pour identifier et corriger les configurations GPU incorrectes. Rechercher systématiquement toute référence à GPU index 0/CUDA:0 et les remplacer par GPU index 1/CUDA:1.

## 🔍 PROTOCOLE D'AUDIT SYSTÉMATIQUE

### ÉTAPE 1 : RECHERCHE EXHAUSTIVE DES PATTERNS INTERDITS
Rechercher dans TOUS les fichiers (.py, .yaml, .json, .md) les patterns suivants **INTERDITS** :

**Patterns GPU Index 0 (INTERDITS)** :
- `gpu_device_index: 0`
- `gpu_device_index=0`
- `gpu_device: "cuda:0"`
- `device="cuda:0"`
- `cuda:0`
- `GPU:0`
- `gpu: 0`
- `device_id=0`
- `gpu_id=0`
- `main_gpu: 0`
- `selected_gpu = 0`
- `torch.cuda.device(0)`
- `.to("cuda:0")`
- `.cuda(0)`

### ÉTAPE 2 : VALIDATION PATTERNS AUTORISÉS
Vérifier que les patterns suivants **AUTORISÉS** sont utilisés :

**Patterns GPU Index 1 (AUTORISÉS)** :
- `gpu_device_index: 1`
- `gpu_device_index=1`
- `gpu_device: "cuda:1"`
- `device="cuda:1"`
- `cuda:1`
- `GPU:1`
- `gpu: 1`
- `device_id=1`
- `gpu_id=1`
- `main_gpu: 1`
- `selected_gpu = 1`
- `torch.cuda.device(1)`
- `.to("cuda:1")`
- `.cuda(1)`

### ÉTAPE 3 : FICHIERS PRIORITAIRES À AUDITER

#### **🔴 TASK 1 & 2 - RobustSTTManager**
- **STT/stt_manager_robust.py** :
  - Ligne ~70 : Logique sélection GPU
  - Ligne ~75 : Monitoring VRAM 
  - Ligne ~173 : Métriques GPU
  - Rechercher : `gpu_count > 1 else 0` (logique fallback incorrecte)
- **tests/test_realtime_audio_pipeline.py** : Configuration tests STT
- **Tous fichiers tests STT** : Vérifier config GPU dans les tests

#### **🔴 TASK 3 - EnhancedLLMManager** 
- **Config/mvp_settings.yaml** :
  - `gpu_device_index` doit être 1
  - `gpu_device` doit être "cuda:1"
- **LLM/llm_manager_enhanced.py** :
  - Défauts GPU (main_gpu, device_id)
  - Initialisation GPU
- **tests/test_llm_handler.py** : Configuration tests LLM
- **tests/test_enhanced_llm_manager.py** : Tous tests LLM
- **tests/demo_enhanced_llm_interface.py** : Interface demo

#### **🔴 TASK 4 - VAD Optimized Manager (en cours)**
- **Vérifier AVANT implémentation** : Aucune config GPU 0 autorisée
- **VAD/vad_manager.py** : Si existant
- **Tests VAD** : Configuration GPU appropriée

#### **🔴 FICHIERS CONFIGURATION GLOBAUX**
- **Config/mvp_settings.yaml** : Configuration principale
- **requirements.txt** : Versions PyTorch avec CUDA
- **Tous .env/.yaml/.json** : Variables GPU
- **Scripts launch** : Commandes CUDA

### ÉTAPE 4 : COMMANDES DE RECHERCHE RECOMMANDÉES

```bash
# Recherche patterns interdits (GPU 0)
grep -r "gpu_device_index.*0" . --include="*.py" --include="*.yaml" --include="*.json"
grep -r "cuda:0" . --include="*.py" --include="*.yaml" --include="*.json"
grep -r "gpu.*0" . --include="*.py" --include="*.yaml" --include="*.json"
grep -r "device.*0" . --include="*.py" --include="*.yaml" --include="*.json"

# Validation patterns autorisés (GPU 1)
grep -r "gpu_device_index.*1" . --include="*.py" --include="*.yaml" --include="*.json"
grep -r "cuda:1" . --include="*.py" --include="*.yaml" --include="*.json"

# Recherche exhaustive tous GPU references
grep -r -i "gpu\|cuda" . --include="*.py" --include="*.yaml" --include="*.json"
```

### ÉTAPE 5 : VALIDATION DU CODE

#### **Vérifications Obligatoires par Fichier** :
1. **Aucune référence GPU index 0** (RTX 5060 interdite)
2. **Toutes références GPU index 1** (RTX 3090 autorisée)
3. **Validation 24GB VRAM** quand disponible
4. **Fallback CPU** si mauvaise détection GPU
5. **Logs explicites** configuration GPU utilisée

#### **Protection Code Requise** :
```python
# EXEMPLE CODE PROTECTION OBLIGATOIRE
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        # Forcer RTX 3090 (index 1)
        device = torch.device("cuda:1")
        # Valider 24GB VRAM (signature RTX 3090)
        vram_gb = torch.cuda.get_device_properties(1).total_memory / 1e9
        if vram_gb < 20:  # RTX 3090 = ~24GB
            print("⚠️ ERREUR: GPU 1 n'est pas RTX 3090 (24GB VRAM)")
            device = torch.device("cpu")  # Fallback sécurisé
    else:
        print("🚨 CRITIQUE: RTX 3090 (GPU 1) non détectée")
        device = torch.device("cpu")  # Fallback obligatoire
```

## 📋 RAPPORT D'AUDIT ATTENDU

### **Format de Sortie Requis** :
```markdown
## RAPPORT AUDIT GPU - SuperWhisper V6

### FICHIERS ANALYSÉS
- Total fichiers scannés : X
- Fichiers avec GPU config : Y
- Fichiers problématiques : Z

### ERREURS TROUVÉES
#### Fichier : path/to/file.py
- Ligne X : `gpu_device_index: 0` ❌ → `gpu_device_index: 1` ✅
- Ligne Y : `cuda:0` ❌ → `cuda:1` ✅

### CORRECTIONS APPLIQUÉES  
- [x] Config/mvp_settings.yaml : RTX 3090 configuré
- [x] STT/stt_manager_robust.py : GPU 1 forcé
- [x] LLM/llm_manager_enhanced.py : défauts corrigés
- [ ] Autres fichiers à corriger...

### VALIDATION FINALE
- ✅ Aucune référence GPU 0 restante
- ✅ Toutes configs pointent GPU 1 (RTX 3090)
- ✅ Protection fallback CPU implémentée
- ✅ Validation 24GB VRAM ajoutée
```

## 🚨 CRITÈRES DE SUCCÈS

### **AUDIT RÉUSSI SI** :
- ✅ **ZÉRO référence GPU index 0** dans tout le projet
- ✅ **TOUTES configurations GPU index 1** (RTX 3090)
- ✅ **Protection critique** : Validation 24GB VRAM implémentée
- ✅ **Fallback sécurisé** : CPU si mauvaise détection GPU
- ✅ **Logging explicite** : Configuration GPU tracée

### **VALIDATION OBLIGATOIRE POST-AUDIT** :
- [ ] Tests runtime avec RTX 3090 configurée
- [ ] Métriques VRAM exclusivement sur GPU 1
- [ ] Aucun warning/erreur GPU dans logs
- [ ] Performance optimale avec 24GB VRAM

## ⚠️ ATTENTION CRITIQUE

**CET AUDIT EST BLOQUANT** : Aucun développement Task 4+ autorisé tant que la configuration GPU n'est pas 100% validée sur RTX 3090 exclusivement.

**SÉCURITÉ MATÉRIELLE** : L'utilisation accidentelle de RTX 5060 (port principal) peut endommager le hardware. La validation de cette configuration est OBLIGATOIRE avant tout test runtime. 