# 🚀 **PROMPT TRANSMISSION PHASE 4 STT - SUPERWHISPER V6**

**Date de transmission** : 12 Juin 2025 - 17:15  
**Phase** : 4 - Speech-to-Text (STT) Integration  
**Statut** : 🎯 **PRÊT DÉMARRAGE JOUR 1**  
**Mission** : Intégration Prism_Whisper2 avec pipeline voix-à-voix complet  

---

## 🎯 **MISSION IMMÉDIATE - PHASE 4 STT**

### **🔴 OBJECTIF PRINCIPAL**
Intégrer **Prism_Whisper2** comme backend STT principal pour créer un **pipeline voix-à-voix complet** (STT → LLM → TTS) avec **performance < 1.2s latence totale** et **validations humaines obligatoires** pour tous tests audio microphone.

### **🚨 RÈGLES ABSOLUES CRITIQUES**
1. **GPU RTX 3090 EXCLUSIF** : CUDA:1 uniquement, RTX 5060 STRICTEMENT INTERDITE
2. **Validations humaines OBLIGATOIRES** : Tous tests audio microphone nécessitent écoute manuelle
3. **Documentation continue** : Journal + suivi tâches, JAMAIS de suppression
4. **Configuration GPU** : Template V2.0 obligatoire dans tous scripts

---

## 📋 **DOCUMENTS CRITIQUES À CONSULTER**

### **🔴 PRIORITÉ ABSOLUE (À lire en PREMIER)**
1. **`docs/suivi_stt_phase4.md`** ✅ **CRÉÉ** - Planning détaillé 3 jours + template validation audio
2. **`docs/prompt.md`** ✅ **MIS À JOUR** - Prompt principal avec validations humaines
3. **`docs/prd.md`** ✅ **MIS À JOUR** - Exigences avec validation humaine obligatoire
4. **`docs/dev_plan.md`** ✅ **MIS À JOUR** - Plan développement avec validations humaines
5. **`docs/ON_BOARDING_ia.md`** - Briefing complet projet (contexte Phase 3 TTS)

### **🟠 PRIORITÉ HAUTE (Contexte technique)**
6. **`.cursorrules`** - Règles GPU RTX 3090 obligatoires
7. **`docs/standards_gpu_rtx3090_definitifs.md`** - Standards GPU absolus
8. **`TTS/tts_manager.py`** - Architecture TTS réussie (inspiration STT Manager)
9. **`tests/test_tts_manager_integration.py`** - Suite tests pytest (modèle pour STT)

---

## 🎮 **CONFIGURATION GPU RTX 3090 - TEMPLATE OBLIGATOIRE**

### **🚨 TEMPLATE V2.0 - À COPIER DANS TOUS SCRIPTS STT**
```python
#!/usr/bin/env python3
"""
SuperWhisper V6 - Phase 4 STT
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 SuperWhisper V6 Phase 4 STT - Configuration GPU RTX 3090 (CUDA:1) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Validation obligatoire RTX 3090
def validate_rtx3090_stt():
    """Validation systématique RTX 3090 pour STT"""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise pour STT")
    
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory < 20:  # RTX 3090 = ~24GB
        raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
    
    print(f"✅ RTX 3090 validée pour STT: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

# APPELER OBLIGATOIREMENT
if __name__ == "__main__":
    validate_rtx3090_stt()
    # Votre code STT ici...
```

---

## 🧪 **PROTOCOLE VALIDATION HUMAINE AUDIO - OBLIGATOIRE**

### **🚨 RÈGLE ABSOLUE**
**TOUS les tests audio microphone nécessitent validation humaine par écoute manuelle.**

### **📋 Template Validation Audio (À utiliser pour chaque test)**
```markdown
## 🎤 VALIDATION HUMAINE AUDIO STT - [DATE/HEURE]

### **Informations Test**
- **Testeur** : [Nom]
- **Date/Heure** : [Date complète]
- **Version STT** : [Version]
- **Backend utilisé** : [Prism_Whisper2/Fallback]
- **Configuration GPU** : RTX 3090 CUDA:1

### **Test Audio Microphone**
- **Phrase testée** : "[Phrase exacte prononcée]"
- **Durée audio** : [X.X secondes]
- **Qualité microphone** : [Bonne/Moyenne/Faible]
- **Environnement** : [Silencieux/Bruyant/Normal]

### **Résultat STT**
- **Transcription obtenue** : "[Texte exact retourné par STT]"
- **Précision** : [Excellent/Bon/Moyen/Faible]
- **Latence perçue** : [< 500ms / 500ms-1s / > 1s]
- **Erreurs détectées** : [Liste des erreurs]

### **Validation Humaine**
- **✅ ACCEPTÉ** / **❌ REFUSÉ**
- **Commentaires** : [Observations détaillées]
- **Actions requises** : [Si refusé, actions correctives]

### **Métriques Techniques**
- **Latence STT** : [XXX ms]
- **Confiance modèle** : [XX%]
- **Utilisation GPU** : [XX%]
- **Mémoire VRAM** : [XX GB]
```

---

## 📊 **CONTEXTE PROJET - SUCCÈS PHASE 3 TTS**

### **🏆 Performance Exceptionnelle Héritée**
- **Latence Cache TTS** : 29.5ms (record absolu, +340% objectif)
- **Taux Cache TTS** : 93.1% (excellent, +116% objectif)
- **Stabilité TTS** : 100% (zéro crash)
- **Tests TTS** : 88.9% succès (très bon)

### **🔧 Infrastructure Réutilisable**
- **UnifiedTTSManager** : Architecture 4 backends avec fallback intelligent
- **Cache LRU** : Ultra-performant 200MB, TTL 2h
- **Tests Pytest** : Suite complète 9 tests
- **Configuration GPU** : Standards RTX 3090 validés
- **Monitoring** : Métriques temps réel

### **🎯 Objectif Phase 4**
**Atteindre le même niveau d'excellence pour STT que celui obtenu en Phase 3 TTS**

---

## 🚀 **ACTIONS IMMÉDIATES JOUR 1 - PRIORITÉ CRITIQUE**

### **🔴 PRIORITÉ ABSOLUE (0-2h)**
1. **Cloner Prism_Whisper2** : `git clone https://github.com/KaizenCoder/Prism_whisper2`
2. **Analyser architecture** : Structure code, dépendances, modèles
3. **Tester RTX 3090** : Validation configuration CUDA:1 exclusive
4. **PoC basique** : Premier test STT simple avec validation humaine

### **🟠 PRIORITÉ HAUTE (2-4h)**
5. **Architecture STTManager** : Design inspiré TTSManager (4 backends)
6. **Backends STT** : PrismSTTBackend + 3 fallbacks
7. **Tests validation** : Premier protocole audio microphone
8. **Documentation** : Architecture STT détaillée

### **🟡 PRIORITÉ MOYENNE (4-8h)**
9. **Intégration TTS** : Coexistence STT + TTS existant
10. **Cache STT** : Design intelligent inspiré cache TTS
11. **Tests automatisés** : Suite pytest STT
12. **Monitoring** : Métriques temps réel STT

---

## 📋 **PLANNING DÉTAILLÉ 3 JOURS**

### **🚀 JOUR 1 - RECHERCHE ET ARCHITECTURE (EN COURS)**
- ✅ **Prism_Whisper2** : Clone, analyse, test RTX 3090
- ✅ **PoC STT** : Premier test avec validation humaine
- ✅ **Architecture** : STTManager design + backends
- ✅ **Documentation** : Architecture détaillée

### **🔧 JOUR 2 - IMPLÉMENTATION CORE (PLANIFIÉ)**
- ✅ **STTManager** : Implémentation 4 backends
- ✅ **Intégration** : Prism_Whisper2 optimisé RTX 3090
- ✅ **Pipeline** : STT→TTS bidirectionnel
- ✅ **Tests** : Suite pytest + validation humaine

### **🧪 JOUR 3 - TESTS ET VALIDATION (PLANIFIÉ)**
- ✅ **Tests intégration** : STT + TTS complets
- ✅ **Performance** : Pipeline < 1.2s latence totale
- ✅ **Validation finale** : Tests humains obligatoires
- ✅ **Livraison** : Code production + documentation

---

## 🎯 **OBJECTIFS MÉTRIQUES PHASE 4**

### **📊 Performance STT**
| Métrique | Objectif | Critique |
|----------|----------|----------|
| **Latence STT** | < 800ms | < 1000ms |
| **Précision** | > 95% | > 90% |
| **Pipeline Total** | < 1.2s | < 1.5s |
| **Stabilité** | > 95% | > 90% |
| **Cache Hit Rate** | > 80% | > 60% |

### **📊 Techniques**
| Composant | Objectif | Statut |
|-----------|----------|---------|
| **Prism_Whisper2** | Intégré | ⏳ À faire |
| **Fallback Multi** | 4 backends | ⏳ À faire |
| **Tests Pytest** | > 85% succès | ⏳ À faire |
| **Validation Humaine** | 100% tests audio | ⏳ À faire |
| **Documentation** | Complète | ⏳ À faire |

---

## 🔧 **ARCHITECTURE STT RECOMMANDÉE**

### **🏗️ Structure Inspirée TTS Manager**
```
STT/
├── stt_manager.py              # UnifiedSTTManager (inspiré TTSManager)
├── backends/
│   ├── prism_stt_backend.py    # Backend principal Prism_Whisper2
│   ├── whisper_direct_backend.py # Fallback 1: faster-whisper direct
│   ├── whisper_cpu_backend.py   # Fallback 2: CPU whisper
│   └── offline_stt_backend.py   # Fallback 3: Windows Speech API
├── utils/
│   ├── audio_utils.py          # Validation audio, preprocessing
│   └── cache_manager.py        # Cache LRU STT (inspiré TTS)
└── config/
    └── stt.yaml               # Configuration STT
```

### **🔄 Backends STT Multi-Niveaux**
1. **PrismSTTBackend** (Principal) - Prism_Whisper2, RTX 3090, 4.5s
2. **WhisperDirectBackend** (Fallback 1) - faster-whisper direct, RTX 3090, 6-7s
3. **WhisperCPUBackend** (Fallback 2) - whisper CPU, 15-20s
4. **OfflineSTTBackend** (Urgence) - Windows Speech API, 2-3s

---

## 📝 **DOCUMENTATION OBLIGATOIRE**

### **📋 Règles Documentation Continue**
1. **Journal développement** : `JOURNAL_DEVELOPPEMENT.md` - Mise à jour obligatoire, JAMAIS de suppression
2. **Suivi tâches** : `docs/suivi_stt_phase4.md` - Mise à jour continue
3. **Validation humaine** : Template obligatoire pour chaque test audio
4. **Métriques** : Suivi temps réel performance
5. **Architecture** : Documentation technique détaillée

### **📁 Fichiers à Maintenir**
- `JOURNAL_DEVELOPPEMENT.md` : Chronologie complète (modification uniquement)
- `docs/suivi_stt_phase4.md` : Suivi Phase 4 (mise à jour continue)
- `docs/prompt.md` : Prompt principal (mise à jour si nécessaire)
- `docs/prd.md` : Exigences projet (mise à jour si nécessaire)
- `docs/dev_plan.md` : Plan développement (mise à jour si nécessaire)

---

## ⚠️ **POINTS CRITIQUES À RETENIR**

### **🚨 RÈGLES ABSOLUES**
- **RTX 5060 = STRICTEMENT INTERDITE** (CUDA:0, 8GB insuffisant)
- **RTX 3090 = SEULE GPU AUTORISÉE** (CUDA:1, 24GB optimal)
- **Template V2.0 = OBLIGATOIRE** pour tous scripts STT
- **Validation humaine = CRITIQUE** pour tous tests audio microphone
- **Documentation continue = OBLIGATOIRE** sans suppression

### **🎯 FOCUS IMMÉDIAT**
1. **Prism_Whisper2** : Analyse et intégration prioritaire
2. **RTX 3090** : Configuration et validation exclusive
3. **PoC STT** : Premier test avec validation humaine
4. **Architecture** : STTManager design robuste

---

## 🎊 **ÉTAT PROJET - PRÊT PHASE 4**

### **✅ ACQUIS SOLIDES**
- **Phase 3 TTS** : Succès exceptionnel (29.5ms latence)
- **Infrastructure** : Tests, monitoring, cache ultra-performant
- **Configuration GPU** : Standards RTX 3090 validés
- **Documentation** : Système professionnel automatisé

### **🚀 PRÊT POUR PHASE 4**
- **Documents** : 4 fichiers mis à jour avec validations humaines
- **Planning** : 3 jours détaillés avec actions prioritaires
- **Architecture** : Design STT inspiré succès TTS
- **Objectifs** : Métriques claires et atteignables

---

## 🎯 **INSTRUCTION FINALE POUR NOUVEAU CHAT**

**COMMENCER IMMÉDIATEMENT PAR :**

1. **Lire** `docs/suivi_stt_phase4.md` (planning détaillé)
2. **Cloner** Prism_Whisper2 : `git clone https://github.com/KaizenCoder/Prism_whisper2`
3. **Analyser** architecture et dépendances
4. **Tester** configuration RTX 3090 CUDA:1
5. **Créer** premier PoC STT avec validation humaine obligatoire

**OBJECTIF JOUR 1 :** PoC STT fonctionnel avec Prism_Whisper2 sur RTX 3090 et première validation humaine audio réussie.

---

*Prompt Transmission Phase 4 STT - SuperWhisper V6*  
*Assistant IA Claude - Anthropic*  
*12 Juin 2025 - 17:15*  
*🚀 PRÊT DÉMARRAGE IMMÉDIAT PHASE 4 STT !* 