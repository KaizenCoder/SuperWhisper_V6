# 🚀 GUIDE RAPIDE - VALIDATION MICROPHONE LIVE

**⏰ Temps estimé** : 15-30 minutes  
**🎯 Objectif** : Valider correction VAD en conditions réelles  
**📊 État** : Correction VAD réussie sur fichier (+492%), validation microphone manquante  

---

## ⚡ DÉMARRAGE RAPIDE (5 minutes)

### **1. Ouvrir Terminal dans SuperWhisper V6**
```bash
cd C:\Dev\SuperWhisper_V6
```

### **2. Lancer Script de Validation**
```bash
python scripts/validation_microphone_live_equipe.py
```

### **3. Suivre Instructions à l'Écran**
- ✅ Validation GPU RTX 3090 automatique
- ✅ Test microphone automatique  
- ✅ Lecture texte fourni au microphone
- ✅ Validation humaine guidée

---

## 📝 TEXTE À LIRE (Fourni par le Script)

Le script affichera automatiquement le texte à lire. **Lisez-le clairement et complètement** au microphone quand demandé.

**Durée lecture** : ~30-45 secondes  
**Mots attendus** : ~138 mots  
**Objectif** : Aucune interruption prématurée de transcription  

---

## 🎯 CRITÈRES DE VALIDATION

### **✅ Succès Attendu**
- **Couverture** : >95% des mots transcrits
- **Précision** : >90% des mots corrects  
- **Pas d'interruption** : Transcription complète
- **Latence** : <15 secondes acceptable

### **🔍 Points de Vigilance**
- Transcription s'arrête prématurément = ❌ ÉCHEC
- Mots manqués massivement = ❌ ÉCHEC  
- Qualité audio dégradée = 🟡 ATTENTION

---

## 📊 RÉSULTAT ATTENDU

### **🎊 Si Validation Réussie**
```
✅ VALIDATION MICROPHONE LIVE RÉUSSIE!
✅ La correction VAD fonctionne parfaitement en conditions réelles
🚀 Phase 4 STT peut être marquée comme TERMINÉE
```

### **⚠️ Si Problèmes Détectés**
```
⚠️ VALIDATION MICROPHONE LIVE PARTIELLE
🔧 Des ajustements peuvent être nécessaires
```

---

## 🔧 RÉSOLUTION PROBLÈMES RAPIDES

### **❌ Erreur GPU**
```bash
# Vérifier RTX 3090
nvidia-smi
python test_gpu_correct.py
```

### **❌ Erreur Microphone**
- Vérifier branchement microphone
- Tester avec autre application (ex: Audacity)
- Ajuster volume microphone Windows

### **❌ Erreur Import STT**
```bash
# Vérifier environnement
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 📋 RAPPORT AUTOMATIQUE

Le script génère automatiquement :
- **Fichier JSON** : `validation_reports/validation_microphone_live_YYYYMMDD_HHMMSS.json`
- **Métriques détaillées** : Couverture, précision, latence
- **Validation humaine** : Feedback utilisateur
- **Décision finale** : VALIDÉ / À CORRIGER / VALIDÉ AVEC RÉSERVES

---

## 📞 SUPPORT

### **Documentation Complète**
- `docs/prompt.md` - Contexte technique complet
- `docs/dev_plan.md` - Plan développement Phase 4
- `docs/prd.md` - Spécifications produit
- `docs/Transmission_Coordinateur/TRANSMISSION_VALIDATION_MICROPHONE_LIVE.md` - Guide détaillé

### **Scripts Alternatifs (si problèmes)**
```bash
# Test microphone simple
python scripts/test_microphone_reel.py

# Diagnostic STT
python scripts/diagnostic_stt_simple.py

# Test correction VAD
python scripts/test_correction_vad_expert.py
```

---

## 🎯 MISSION CRITIQUE

**Cette validation est la DERNIÈRE ÉTAPE** pour compléter la Phase 4 STT SuperWhisper V6.

**Succès = Phase 4 TERMINÉE**  
**Échec = Corrections requises**

---

**🚀 LANCEZ LE SCRIPT ET SUIVEZ LES INSTRUCTIONS !** 