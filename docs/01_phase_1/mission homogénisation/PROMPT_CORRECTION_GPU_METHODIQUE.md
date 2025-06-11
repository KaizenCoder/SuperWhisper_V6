# 🔧 PROMPT MÉTHODIQUE - CORRECTION CONFIGURATION GPU SUPERWHISPER V6

## 🎯 **MISSION CRITIQUE :** Correction et Validation GPU RTX 3090

### **📋 CONTEXTE RÉVÉLÉ :**
Suite à découverte majeure : **GPU 0 = RTX 3090 (24GB) ✅** / **GPU 1 = RTX 5060 Ti (16GB) ❌**  
Plusieurs fichiers ont été incorrectement modifiés vers GPU 1 au lieu de GPU 0.

---

## 📚 **DOCUMENTS DE RÉFÉRENCE OBLIGATOIRES**

### **📄 Documentation Critique :**
1. `docs/phase_1/DECOUVERTE_CRITIQUE_GPU.md` - Révélation configuration réelle
2. `test_cuda_debug.py` - Script validation GPU définitif
3. `.cursor/rules/cursor_rules.mdc` - Règles GPU obligatoires du projet
4. `docs/phase_1/phase_1_configuration_audit.md` - Audit initial

### **🔍 Scripts de Validation :**
- `test_cuda_debug.py` - Test configuration GPU système
- `validate_gpu_config.py` - Validation patterns dans code
- `test_rtx3090_detection.py` - Test détection RTX 3090

### **⚙️ Fichiers Configuration Système :**
- `Config/mvp_settings.yaml` - Configuration principale projet
- Tous fichiers `*_manager_*.py` - Gestionnaires GPU

---

## 🎯 **OBJECTIFS DE CORRECTION**

### **🚨 PRIORITÉ 1 - CORRECTIONS CRITIQUES :**
1. **Revenir à GPU 0 (RTX 3090)** pour tous les fichiers modifiés
2. **Tester CHAQUE correction** immédiatement après application
3. **Valider GPU utilisée** avec preuves factuelles (nom + VRAM)
4. **Documenter CHAQUE test** avec résultats concrets

### **🔍 PRIORITÉ 2 - VALIDATION SYSTÉMATIQUE :**
- Aucune assumption autorisée - que des faits validés
- Test fonctionnel obligatoire après chaque modification  
- Vérification GPU name = "RTX 3090" ET VRAM = ~24GB
- Logging détaillé de tous les tests

---

## 📂 **FICHIERS À CORRIGER IDENTIFIÉS**

### **🔧 SELON RAPPORT_CORRECTIONS_GPU.MD - 6 CORRECTIONS DOCUMENTÉES :**

#### **CORRECTION 1 : tests/test_stt_handler.py**
- **ERREUR IDENTIFIÉE :** Lignes 24, 75, 77, 415 - Tests STT utilisant RTX 5060 (CUDA:0)
- **CORRECTIONS REQUISES :**
  - Ligne 24: `'gpu_device': 'cuda:0'` → `'gpu_device': 'cuda:1'`
  - Ligne 75: `assert handler.device == 'cuda:0'` → `assert handler.device == 'cuda:1'`
  - Ligne 77: `mock_model_instance.to.assert_called_with('cuda:0')` → `mock_model_instance.to.assert_called_with('cuda:1')`
  - Ligne 415: `'gpu_device': 'cuda:0'` → `'gpu_device': 'cuda:1'`
- **TEST OBLIGATOIRE :** Exécuter tests STT et vérifier GPU RTX 3090 utilisée

#### **CORRECTION 2 : utils/gpu_manager.py**
- **ERREUR IDENTIFIÉE :** Lignes 146, 152 - Fallback par défaut vers RTX 5060 (CUDA:0)
- **CORRECTIONS REQUISES :**
  - Ligne 146: `return "cuda:0"` → `return "cuda:1"` (fallback purpose)
  - Ligne 152: `return "cuda:0"` → `return "cuda:1"` (fallback général)
  - Auto-detection LLM: `gpu_map["llm"] = 0` → `gpu_map["llm"] = 1`
  - Auto-detection STT: `gpu_map["stt"] = 0` → `gpu_map["stt"] = 1`
  - Bug fix: `props.max_threads_per_block` → `getattr(props, 'max_threads_per_block', 1024)`
- **TEST OBLIGATOIRE :** Tester GPU Manager et vérifier fallback vers RTX 3090

#### **CORRECTION 3 : docs/Transmission_coordinateur/.../mvp_settings.yaml**
- **ERREUR IDENTIFIÉE :** Lignes 6, 10 - Configuration legacy pointant RTX 5060
- **CORRECTIONS REQUISES :**
  - Ligne 6: `gpu_device: "cuda:0"` → `gpu_device: "cuda:1"`
  - Ligne 10: `gpu_device_index: 0` → `gpu_device_index: 1`
- **TEST OBLIGATOIRE :** Charger config legacy et vérifier GPU pointée

#### **CORRECTION 4 : STT/stt_manager_robust.py - VULNÉRABILITÉ CRITIQUE**
- **ERREUR IDENTIFIÉE :** Lignes 80, 84, 87, 92 - Fallback vers RTX 5060 en single-GPU
- **CORRECTIONS CRITIQUES REQUISES :**
  - Ligne 80: `selected_gpu = 0` → `selected_gpu = 1` (fallback sécurisé)
  - Ligne 84: `target_gpu = 1 if gpu_count >= 2 else 0` → `target_gpu = 1` (inconditionnel)
  - Ligne 87: Validation VRAM inconditionnelle (supprimer condition dual-GPU)
  - Ligne 92: Confirmation RTX 3090 inconditionnelle (supprimer condition dual-GPU)
- **TEST OBLIGATOIRE :** Tester configurations single ET dual GPU, vérifier RTX 3090 exclusive

#### **CORRECTION 5 : test_tts_rtx3090_performance.py**
- **ERREUR IDENTIFIÉE :** Lignes 59, 60 - Test performance utilisant RTX 5060
- **CORRECTIONS REQUISES :**
  - Ligne 59: `torch.cuda.get_device_name(0)` → `torch.cuda.get_device_name(1)`
  - Ligne 60: `torch.cuda.get_device_properties(0)` → `torch.cuda.get_device_properties(1)`
- **TEST OBLIGATOIRE :** Exécuter test performance et vérifier RTX 3090 détectée

#### **CORRECTION 6 : test_rtx3090_detection.py**
- **ERREUR IDENTIFIÉE :** Lignes 26, 27, 28 - Test détection utilisant RTX 5060
- **CORRECTIONS REQUISES :**
  - Ligne 26: `torch.cuda.get_device_name(0)` → `torch.cuda.get_device_name(1)`
  - Ligne 27: `torch.cuda.get_device_properties(0)` → `torch.cuda.get_device_properties(1)`
  - Ligne 28: `torch.cuda.get_device_capability(0)` → `torch.cuda.get_device_capability(1)`
- **TEST OBLIGATOIRE :** Exécuter test détection et vérifier RTX 3090 identifiée

### **🔧 SELON JOURNAL_DEVELOPPEMENT_UPDATED.MD - FICHIERS SUPPLÉMENTAIRES :**

#### **FICHIERS HISTORIQUES MENTIONNÉS :**
- **Config/mvp_settings.yaml** - Configuration principale projet
- **LLM/llm_manager_enhanced.py** - `device = "cuda:0"` vers RTX 3090
- **tests/test_llm_handler.py** - Tests LLM utilisant mauvaise GPU

### **🔧 FICHIERS ADDITIONNELS POTENTIELS (À VÉRIFIER) :**
- `LUXA_TTS/tts_handler_coqui.py` - Handlers TTS
- `TTS/tts_handler_coqui.py` - Handlers TTS  
- `Orchestrator/fallback_manager.py` - Gestionnaire fallback

---

## ⚡ **MÉTHODOLOGIE DE CORRECTION**

### **🔄 PROCESSUS POUR CHAQUE FICHIER :**

#### **ÉTAPE 1 - ANALYSE PRÉ-CORRECTION :**
```bash
# 1. Lire contenu actuel fichier
# 2. Identifier valeurs GPU actuelles  
# 3. Déterminer corrections nécessaires
# 4. Créer script test spécifique au fichier
```

#### **ÉTAPE 2 - APPLICATION CORRECTION :**
```bash  
# 1. Appliquer modification GPU → 0 (RTX 3090)
# 2. Sauvegarder fichier modifié
# 3. Vérifier syntaxe/format correct
```

#### **ÉTAPE 3 - VALIDATION IMMÉDIATE :**
```python
# TEMPLATE TEST OBLIGATOIRE POUR CHAQUE FICHIER :
import torch
import os

def test_fichier_gpu():
    print("🔍 TEST GPU pour [NOM_FICHIER]")
    
    # Charger/initialiser le fichier modifié
    # [CODE SPÉCIFIQUE SELON FICHIER]
    
    # VALIDATION OBLIGATOIRE
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU détectée: {gpu_name}")
        print(f"✅ VRAM disponible: {gpu_memory:.1f}GB")
        
        # VALIDATION CRITIQUE
        assert "RTX 3090" in gpu_name, f"❌ GPU incorrecte: {gpu_name}"
        assert gpu_memory > 20, f"❌ VRAM insuffisante: {gpu_memory}GB"
        
        print("✅ VALIDATION PASSÉE - RTX 3090 confirmée")
    else:
        raise RuntimeError("❌ CUDA non disponible")

if __name__ == "__main__":
    test_fichier_gpu()
```

#### **ÉTAPE 4 - DOCUMENTATION RÉSULTATS :**
```markdown
## Test [FICHIER] - [TIMESTAMP]
- **Modification appliquée :** [DÉTAIL EXACT]
- **GPU détectée :** [NOM GPU EXACT]  
- **VRAM disponible :** [VALEUR EXACTE]
- **Statut :** ✅ SUCCÈS / ❌ ÉCHEC
- **Preuves :** [LOGS/OUTPUTS EXACTS]
```

---

## 🚫 **ERREURS CRITIQUES À NE PAS REPRODUIRE**

### **❌ INTERDICTIONS ABSOLUES :**
1. **Aucune assumption** sur configuration système
2. **Aucune modification en masse** sans test individuel
3. **Aucune confiance** en documentation sans validation pratique
4. **Aucune correction** sans test immédiat après
5. **Aucune interprétation** de CUDA_VISIBLE_DEVICES sans test

### **❌ PIÈGES SPÉCIFIQUES ÉVITÉS :**
- Croire que `CUDA_VISIBLE_DEVICES='1'` change l'indexation PyTorch
- Appliquer corrections basées sur théorie au lieu de faits
- Ignorer doutes/questionnements sur la logique
- Faire confiance aux patterns sans validation système

---

## 💡 **PISTES D'AMÉLIORATION IDENTIFIÉES**

### **🔧 AMÉLIORATION PROCESSUS :**
1. **Script validation automatique** après chaque modification
2. **Tests d'intégration continus** pour configuration GPU
3. **Alertes automatiques** si GPU incorrecte détectée
4. **Documentation vivante** mise à jour automatiquement

### **🛡️ PRÉVENTION FUTURES ERREURS :**
1. **Template test GPU obligatoire** pour tous nouveaux fichiers
2. **CI/CD check GPU** avant validation commits
3. **Monitoring configuration** en temps réel
4. **Formation équipe** sur validation factuelle

### **📊 MONITORING AVANCÉ :**
```python
# Script monitoring GPU à intégrer :
def monitor_gpu_usage():
    """Monitoring continu utilisation GPU"""
    # Vérification périodique GPU active
    # Alertes si RTX 5060 Ti détectée
    # Logs détaillés utilisation VRAM
    # Dashboard temps réel
```

---

## 📋 **RAPPORT FINAL REQUIS**

### **📊 STRUCTURE RAPPORT OBLIGATOIRE :**

```markdown
# 🎯 RAPPORT CORRECTION GPU - SuperWhisper V6

## 📅 MÉTADONNÉES
- **Date :** [TIMESTAMP]
- **Nombre fichiers corrigés :** [N]
- **Tests effectués :** [N]
- **Durée totale :** [TEMPS]

## ✅ SUCCÈS CONFIRMÉS
### Fichier 1: [NOM]
- **Correction :** [DÉTAIL]
- **GPU Validée :** [NOM + VRAM]
- **Preuve :** [LOG EXACT]

[... RÉPÉTER POUR CHAQUE FICHIER ...]

## ❌ ÉCHECS RENCONTRÉS  
### Fichier X: [NOM]
- **Problème :** [DESCRIPTION]
- **Cause :** [ANALYSE]
- **Solution :** [ACTION REQUISE]

## 🔍 VALIDATIONS FACTUELLES
- **Total tests GPU :** [N] 
- **RTX 3090 confirmations :** [N]
- **VRAM 24GB validations :** [N]
- **Échecs détectés :** [N]

## 📈 RECOMMANDATIONS
1. [AMÉLIORATION 1]
2. [AMÉLIORATION 2]  
3. [AMÉLIORATION 3]

## 🎯 STATUT FINAL
- [ ] Toutes corrections appliquées
- [ ] Tous tests passés  
- [ ] GPU RTX 3090 confirmée partout
- [ ] Documentation mise à jour
```

---

## ⚡ **INSTRUCTIONS D'EXÉCUTION**

### **🚀 ORDRE D'EXÉCUTION STRICT :**
1. **Lire TOUS les documents de référence** mentionnés
2. **Analyser chaque fichier** à corriger individuellement
3. **Appliquer correction** + **Test immédiat** + **Documentation**
4. **Passer au fichier suivant** seulement si test passé
5. **Compiler rapport final** avec toutes les preuves factuelles

### **🔍 VALIDATION CONTINUE :**
- Après chaque modification : test GPU obligatoire
- Après chaque test : documentation résultat
- Avant passage fichier suivant : validation succès
- À la fin : rapport complet avec preuves

### **🎯 CRITÈRES DE SUCCÈS :**
- **100% des fichiers** utilisent RTX 3090 (GPU 0)
- **100% des tests** confirment "RTX 3090" + "~24GB"  
- **0 assumption** - que des validations factuelles
- **Documentation complète** avec preuves pour audit

---

**🚨 RAPPEL CRITIQUE :** Aucune assumption autorisée. Chaque affirmation doit être validée par des faits. Chaque correction doit être testée immédiatement. Chaque test doit être documenté avec preuves concrètes.

**🎯 OBJECTIF FINAL :** Configuration GPU 100% fiable avec RTX 3090 exclusivement, validée par des tests factuels sur chaque fichier du projet. 