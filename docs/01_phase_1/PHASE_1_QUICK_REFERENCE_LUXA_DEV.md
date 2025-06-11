# Guide Référence Rapide - Développement LUXA
## SuperWhisper_V6 - Phase 1 en cours

**Dernière mise à jour**: 2025-01-09  
**État actuel**: ✅ Tâche 2 TERMINÉE, 🎯 Tâche 3 PRÊTE  

---

## 🚀 État Projet Actuel

### Tâches Taskmaster Status
```bash
# Commande vérification rapide
task-master list --with-subtasks

# État actuel:
✅ Tâche 1: Correction Import Bloquant - TERMINÉE
✅ Tâche 2: RobustSTTManager Implementation - TERMINÉE
  ✅ 2.1: Manager Implementation - TERMINÉE  
  ✅ 2.2: Test Script Adaptation - TERMINÉE
  ✅ 2.3: Orchestrator Integration - TERMINÉE
🎯 Tâche 3: EnhancedLLMManager - PRÊTE (complexité 8, priorité haute)
```

### Prochaine Action Immédiate
```bash
# Commencer Tâche 3
task-master next
task-master show 3  # Voir détails complets

# Si breakdown nécessaire  
task-master expand --id=3 --research --prompt="Focus sur context management et performance"
```

---

## 📁 Architecture Fichiers Récents

### Nouveaux Fichiers Créés (2025-01-09)
- `STT/stt_manager_robust.py` - RobustSTTManager production-ready
- `tests/test_realtime_audio_pipeline.py` - Tests validation microphone réel
- `docs/phase_1/ROBUST_STT_MANAGER_SYNTHESIS.md` - Synthèse technique complète

### Fichiers Modifiés
- `run_assistant.py` - Intégration RobustSTTManager + async conversion
- `docs/phase_1/journal_developpement.md` - Entrée complète 2025-01-09

### Configuration Taskmaster
- `.taskmaster/tasks/tasks.json` - Tâches alignées Plan LUXA Final
- `.taskmaster/tasks/PRD_LUXA_Development_Final.txt` - PRD consolidé

---

## 🎯 Workflow Développement Standard

### 1. Début de Session
```bash
# Vérifier état
task-master list
task-master next

# Analyser tâche suivante
task-master show <id>
task-master complexity-report  # Si analyse disponible
```

### 2. Implementation
```bash
# Si tâche complexe, breakdown
task-master expand --id=<id> --research

# Pendant développement - documenter
# Utiliser prompt.md comme source authority
# Implémenter exactement selon spécifications

# Marquer subtâches terminées
task-master set-status --id=<subtask-id> --status=done
```

### 3. Validation
```bash
# Tests selon stratégie définie
pytest -v -s tests/

# Validation finale
task-master set-status --id=<parent-id> --status=done
task-master generate  # Régénérer fichiers markdown
```

### 4. Documentation (OBLIGATOIRE)
- Mettre à jour journal_developpement.md avec nouvelle entrée
- Suivre template établi
- Documenter décisions, constats, préconisations

---

## 🔧 Commandes Utiles Fréquentes

### Taskmaster Management
```bash
# Navigation tâches
task-master list --status=pending
task-master next
task-master show <id>

# Modification statuts
task-master set-status --id=<id> --status=done
task-master set-status --id=<id> --status=in-progress

# Breakdown/Expansion
task-master expand --id=<id> --research --num=<n>
task-master clear-subtasks --id=<id>  # Si regeneration nécessaire

# Maintenance
task-master validate-dependencies
task-master fix-dependencies
task-master generate
```

### Tests et Validation
```bash
# Tests spécifiques récents
pytest -v -s tests/test_realtime_audio_pipeline.py
pytest -v -s tests/test_realtime_audio_pipeline.py::test_robust_stt_manager_validation_complete

# Tests complets
pytest tests/ --verbose
```

### Git Workflow
```bash
# Avant commit - validation
task-master validate-dependencies
pytest tests/

# Commit structuré
git add .
git commit -m "feat(module): Description courte

- Détail changement 1
- Détail changement 2  
- Référence tâche Taskmaster

Closes Task <id> selon Plan LUXA"
```

---

## 📋 Plan de Développement LUXA - Vue d'Ensemble

### Phase 1 - Managers Critiques
- ✅ **Tâche 1**: Correction Import Bloquant  
- ✅ **Tâche 2**: RobustSTTManager + Validation Microphone
- 🎯 **Tâche 3**: EnhancedLLMManager (PROCHAINE)
- ⏳ **Tâche 4**: OptimizedVADManager Enhancement
- ⏳ **Tâche 5**: UnifiedTTSManager  
- ⏳ **Tâche 6**: AudioStreamManager
- ⏳ **Tâche 7**: ConversationContextManager
- ⏳ **Tâche 8**: MasterOrchestratorEnhanced
- ⏳ **Tâche 9**: Integration Testing Complete
- ⏳ **Tâche 10**: Performance Optimization Final

### Approche Standard
1. **Authority Source**: `prompt.md` pour toutes implémentations
2. **Validation Continue**: Tests conditions réelles obligatoires
3. **Production Ready**: Monitoring, fallbacks, robustesse
4. **Documentation**: Journal obligatoire chaque session

---

## 🚨 Points Critiques à Retenir

### Conformité Plan LUXA
- **Séquentiel strict**: Une tâche complète avant passage suivante
- **Validation obligatoire**: Tests conditions réelles (microphone, etc.)
- **Source unique**: prompt.md pour toutes spécifications
- **Robustesse**: Production-ready dès implémentation

### Architecture Critique
- **VAD Manager**: OptimizedVADManager existant à préserver
- **GPU Management**: Intelligence sélection + fallback CPU
- **Monitoring**: Prometheus metrics systématiques
- **Async Pipeline**: Support asynchrone obligatoire

### Tests Validation
- **Microphone réel**: Tests synthétiques insuffisants
- **Performance**: Mesures latence contre objectifs PRD
- **Robustesse**: Tests fallback et conditions dégradées
- **Integration**: Tests complets pipeline end-to-end

---

## 📊 Métriques Projet Actuelles

### Code Stats
- **Fichiers STT**: 4 modules (MVP + VAD + Benchmark + **Robust**)
- **Tests**: 6 modules complets
- **Coverage**: >80% avec tests conditions réelles
- **Architecture**: Production-ready avec monitoring

### Performance Targets (PRD v3.1)
- **STT Latency**: <300ms audio court
- **VAD Response**: <25ms détection  
- **VRAM Usage**: <80% maximum
- **Error Rate**: <1% transcriptions

### Taskmaster Health
- **Dependencies**: Validées et cohérentes
- **Configuration**: Alignée Plan LUXA 100%
- **Documentation**: Journal à jour complet
- **Status**: Prêt pour Tâche 3

---

## 🔮 Prochaines Étapes Immédiates

### Avant Démarrage Tâche 3
1. **Runtime Validation**: Exécuter tests microphone RobustSTTManager
2. **Dependencies Check**: Vérifier installations complètes
3. **Performance Baseline**: Mesures latence actuelle

### Démarrage Tâche 3 - EnhancedLLMManager
1. **Analyse**: `task-master show 3` pour détails complets
2. **Breakdown**: `task-master expand --id=3 --research` si nécessaire
3. **Implementation**: Suivre prompt.md exclusivement
4. **Tests**: Conversations multi-tours + context management

### Maintenance Continue
1. **Journal**: Documenter chaque session développement
2. **Taskmaster**: Maintenir statuts à jour
3. **Git**: Commits structurés avec références tâches
4. **Tests**: Validation continue conditions réelles

---

**Le projet LUXA suit parfaitement le Plan de Développement Final. L'architecture robuste est en place. Prêt pour EnhancedLLMManager - Tâche 3.** 