# 📋 PRD - CONSOLIDATION TTS SUPERWHISPER V6 (PHASE 2 ENTERPRISE)

**Date :** 2025-06-12  
**Version :** v2.0 Enterprise  
**Statut :** Approuvé pour implémentation  
**Équipe :** SuperWhisper V6 Core Team  

---

## 🎯 **EXECUTIVE SUMMARY**

### **Problème Business :**
Le sous-système TTS de SuperWhisper V6 souffre d'une **fragmentation critique** avec 15 handlers redondants, causant une **complexité de maintenance insoutenable** et des **risques d'instabilité**. La performance n'est pas garantie (<1000ms vs objectif <120ms) et l'architecture actuelle ne permet pas de scaling enterprise.

### **Solution Proposée :**
Implémentation d'une **architecture UnifiedTTSManager enterprise-grade** avec 4 handlers hiérarchisés, circuit breakers, cache intelligent et monitoring Prometheus, garantissant une **performance <120ms** et une **disponibilité 99.9%**.

### **Impact Business :**
- **Réduction complexité** : -87% de fichiers TTS (15→4 handlers)
- **Performance garantie** : <120ms latence (vs <1000ms actuel)
- **Disponibilité enterprise** : 99.9% via fallback automatique
- **Maintenabilité** : Architecture modulaire + monitoring intégré
- **Expérience utilisateur** : Réactivité vocale optimale pour assistant IA

---

## 📊 **CONTEXTE PROJET**

### **🏗️ Architecture SuperWhisper V6 :**
```
🎤 MICROPHONE → STT (Whisper) → LLM (Llama) → TTS (Piper) → 🔊 SPEAKERS
                    ↓               ↓              ↓
                VAD Manager    Context Manager   Audio Output
```

### **🚨 Configuration Hardware Critique :**
- **RTX 5060 (8GB) CUDA:0** ❌ **STRICTEMENT INTERDITE**
- **RTX 3090 (24GB) CUDA:1** ✅ **SEULE GPU AUTORISÉE**
- **RAM :** 64GB DDR4-4800
- **CPU :** Intel Core Ultra 7 265K (20 threads)
- **Stockage Modèles :** `D:\TTS_Voices\` ✅ **EXCLUSIVEMENT**

### **📈 État Actuel :**
- **Modules fonctionnels :** 6/18 (33%)
- **TTS handlers :** 15 fragmentés (2 fonctionnels)
- **Performance TTS :** <1000ms (CLI) vs objectif <120ms
- **Disponibilité :** 95% (fallback basique)
- **Modèles disponibles :** `D:\TTS_Voices\piper\` (fr_FR-siwis-medium.onnx 63MB)

---

## 🎯 **OBJECTIFS QUANTIFIABLES**

### **📊 KPIs Performance :**
| Métrique | Baseline Actuel | Objectif Cible | Amélioration |
|----------|-----------------|----------------|--------------|
| **Latence TTS Principal** | <1000ms (CLI) | <120ms (GPU) | **88% plus rapide** |
| **Latence Fallback 1** | <1000ms | <1000ms | Maintenu |
| **Latence Fallback 2** | N/A | <2000ms | Nouveau |
| **Disponibilité** | 95% | 99.9% | **+4.9%** |
| **Handlers TTS** | 15 fragmentés | 4 unifiés | **-73% complexité** |

### **📊 KPIs Robustesse :**
| Métrique | Baseline | Cible | Mesure |
|----------|----------|-------|--------|
| **MTBF** (Mean Time Between Failures) | 24h | 168h | Semaine sans panne |
| **MTTR** (Mean Time To Recovery) | 30s | <5s | Fallback automatique |
| **Cache Hit Rate** | 0% | >80% | Phrases récurrentes |
| **Circuit Breaker Efficiency** | N/A | >95% | Isolation pannes |

---

## 📋 **EXIGENCES FONCTIONNELLES**

### **FR1 - Interface Unifiée :**
**Description :** Le système doit exposer une unique méthode asynchrone pour toutes les opérations TTS.
```python
async def synthesize(
    text: str, 
    voice: Optional[str] = None,
    speed: Optional[float] = None, 
    reuse_cache: bool = True
) -> TTSResult
```
**Critères d'acceptation :**
- ✅ Interface unique pour tous les backends
- ✅ Retour standardisé `TTSResult`
- ✅ Support paramètres optionnels (voice, speed)
- ✅ Cache configurable par appel

### **FR2 - Fallback Automatique 4 Niveaux :**
**Description :** En cas d'échec d'un handler, basculement transparent vers le niveau suivant.
```
Niveau 1: PiperNativeHandler (GPU) → <120ms
Niveau 2: PiperCliHandler (CPU) → <1000ms  
Niveau 3: SapiFrenchHandler (SAPI) → <2000ms
Niveau 4: SilentEmergencyHandler → <5ms
```
**Critères d'acceptation :**
- ✅ Basculement automatique sans intervention
- ✅ Ordre de priorité respecté
- ✅ Logging détaillé des basculements
- ✅ Métriques de fallback exportées

### **FR3 - Circuit Breaker Pattern :**
**Description :** Isolation automatique des handlers défaillants pour éviter la surcharge.
**Paramètres :**
- **Seuil d'échec :** 3 échecs consécutifs
- **Timeout isolation :** 30 secondes
- **États :** Fermé → Ouvert → Semi-ouvert
**Critères d'acceptation :**
- ✅ Isolation après 3 échecs
- ✅ Réinitialisation automatique après 30s
- ✅ Monitoring état circuit breakers
- ✅ Logs détaillés transitions d'état

### **FR4 - Cache Intelligent :**
**Description :** Cache LRU pour les synthèses fréquentes avec TTL.
**Paramètres :**
- **Taille max :** 100MB
- **TTL :** 1 heure
- **Politique :** LRU (Least Recently Used)
**Critères d'acceptation :**
- ✅ Cache hit <5ms
- ✅ Éviction LRU automatique
- ✅ TTL respecté
- ✅ Métriques cache (hit rate, size)

### **FR5 - Configuration Externalisée :**
**Description :** Tous les paramètres gérés via fichier YAML centralisé.
**Fichier :** `config/tts.yaml`
**Critères d'acceptation :**
- ✅ Aucune valeur codée en dur
- ✅ Rechargement à chaud possible
- ✅ Validation schéma YAML
- ✅ Valeurs par défaut sécurisées

---

## 📋 **EXIGENCES NON-FONCTIONNELLES**

### **NFR1 - Performance :**
- **Latence P95 :** <120ms (PiperNative), <1000ms (PiperCLI), <2000ms (SAPI)
- **Throughput :** >10 synthèses/seconde
- **VRAM GPU :** ≤10% RTX 3090 (90% réservé LLM)
- **CPU Usage :** <20% pendant synthèse

### **NFR2 - Disponibilité :**
- **Uptime :** 99.9% (8.76h downtime/an max)
- **Fallback :** <5s basculement automatique
- **Recovery :** Automatique sans intervention
- **Monitoring :** Alertes temps réel

### **NFR3 - Scalabilité :**
- **Concurrence :** 5 synthèses simultanées
- **Memory :** <500MB RAM total
- **Storage :** <200MB cache max
- **Network :** 0 (100% local)

### **NFR4 - Sécurité :**
- **Validation input :** Sanitization texte
- **Isolation :** Handlers sandboxés
- **Logs :** Pas de données sensibles
- **Access :** Interface interne uniquement

### **NFR5 - Maintenabilité :**
- **Type hints :** 100% coverage
- **Documentation :** Docstrings complètes
- **Tests :** >90% coverage
- **Monitoring :** Métriques Prometheus

### **NFR6 - Stockage & Modèles :**
- **Répertoire obligatoire :** `D:\TTS_Voices\` exclusivement
- **Modèles disponibles :** Vérification préalable avant téléchargement
- **Interdiction absolue :** Stockage modèles ailleurs que sur D:\
- **Modèles validés :** fr_FR-siwis-medium.onnx (63MB) + .json

### **NFR7 - Validation Pratique :**
- **Tests réels obligatoires :** Génération fichiers audio pour écoute
- **Validation manuelle :** Qualité voix française acceptable
- **Benchmark performance :** 10 mesures par cas avec statistiques P95
- **Tests fallback :** Simulation pannes avec validation audio

---

## 🏗️ **ARCHITECTURE TECHNIQUE**

### **🎯 Composants Principaux :**

#### **1. UnifiedTTSManager :**
```python
class UnifiedTTSManager:
    """Orchestrateur principal avec fallback automatique"""
    - Circuit breakers par handler
    - Cache LRU intelligent  
    - Monitoring Prometheus
    - Configuration YAML
```

#### **2. Handlers Hiérarchisés :**
```python
# Niveau 1 - Performance optimale
class PiperNativeHandler(TTSHandler):
    """GPU RTX 3090, <120ms, piper-python native"""

# Niveau 2 - Fallback robuste  
class PiperCliHandler(TTSHandler):
    """CPU subprocess, <1000ms, piper.exe CLI"""

# Niveau 3 - Fallback système
class SapiFrenchHandler(TTSHandler):
    """Windows SAPI, <2000ms, voix française"""

# Niveau 4 - Sécurité ultime
class SilentEmergencyHandler(TTSHandler):
    """Silence généré, <5ms, évite crash"""
```

#### **3. Composants Support :**
```python
class CircuitBreaker:
    """Isolation handlers défaillants"""

class TTSCache:
    """Cache LRU avec TTL"""

class PrometheusMetrics:
    """Métriques temps réel"""
```

### **🔧 Configuration YAML :**
```yaml
# config/tts.yaml
backends:
  piper_native:
    enabled: true
    model_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
    model_config_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx.json"
    device: "cuda:0"  # Pointera RTX 3090 après CUDA_VISIBLE_DEVICES
    target_latency_ms: 120

  piper_cli:
    enabled: true
    model_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
    executable_path: "piper/piper.exe"
    target_latency_ms: 1000

cache:
  enabled: true
  max_size_mb: 100
  ttl_seconds: 3600

circuit_breaker:
  failure_threshold: 3
  reset_timeout_seconds: 30

monitoring:
  prometheus_enabled: true
  prometheus_port: 9090
```

---

## 🧪 **STRATÉGIE DE TEST**

### **📊 Types de Tests :**

#### **1. Tests Unitaires :**
- **Chaque handler** individuellement
- **Composants** (CircuitBreaker, Cache)
- **Configuration** YAML loading
- **Coverage :** >90%

#### **2. Tests Intégration :**
- **Fallback automatique** (simulation pannes)
- **Circuit breakers** (seuils déclenchement)
- **Cache** (hit/miss scenarios)
- **Configuration** end-to-end

#### **3. Tests Performance :**
- **Benchmarks latence** <120ms validation
- **Load testing** concurrence
- **Memory profiling** VRAM/RAM
- **Stress testing** stabilité

#### **4. Tests Régression :**
- **Comparaison avant/après** consolidation
- **Audio quality** (hash comparison)
- **Performance baseline** maintenue
- **Fonctionnalités** préservées

#### **5. Tests Réels Pratiques :**
- **Génération audio** : 4 fichiers test pour écoute manuelle
- **Validation qualité** : Voix française compréhensible
- **Benchmark performance** : 10 mesures par cas (P95 validation)
- **Test fallback** : Simulation pannes avec audio généré
- **Scripts validation** : test_tts_real.py, test_fallback_real.py, test_performance_real.py

### **🎯 Environnements Test :**
- **Local :** Développement + debugging
- **Staging :** Validation pré-production
- **Production :** Monitoring continu

---

## 📅 **PLANNING DÉVELOPPEMENT**

### **🕒 Timeline (5.5 jours) :**

#### **Phase 0 - Préparation (0.5 jour) :**
- ✅ Branche feature + tag sauvegarde
- ✅ Archivage 13 handlers obsolètes
- ✅ Script rollback automatisé

#### **Phase 1 - PiperNativeHandler (2 jours) :**
- 🔧 Diagnostic handler défaillant
- 🔧 Réparation intégration GPU
- 🔧 Validation <120ms
- 🔧 Tests performance

#### **Phase 2 - UnifiedTTSManager (2 jours) :**
- 🔧 Implémentation manager principal
- 🔧 Circuit breakers + cache
- 🔧 Configuration YAML
- 🔧 Tests unitaires + intégration

#### **Phase 3 - Déploiement (1 jour) :**
- 🔧 Feature flags activation
- 🔧 Monitoring Prometheus
- 🔧 Tests validation complète
- 🔧 Documentation + rollback

### **🎯 Jalons Critiques :**
- **J2 :** PiperNativeHandler <120ms validé
- **J4 :** UnifiedTTSManager fonctionnel
- **J5.5 :** Déploiement production ready

---

## ⚠️ **RISQUES ET MITIGATION**

### **🚨 Risques Techniques :**

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| **PiperNativeHandler échec** | CRITIQUE | Moyen | Fallback architecture actuelle |
| **Performance <120ms non atteinte** | ÉLEVÉ | Faible | Optimisation GPU + profiling |
| **Régression fonctionnelle** | ÉLEVÉ | Faible | Tests exhaustifs + rollback |
| **Dépendances manquantes** | MOYEN | Moyen | Validation environnement |

### **🛡️ Stratégies Mitigation :**
- **Checkpoints bloquants** à chaque phase
- **Rollback automatisé** si échec critique
- **Tests régression** systématiques
- **Monitoring continu** post-déploiement

---

## 🎖️ **CRITÈRES D'ACCEPTATION**

### **✅ Critères Fonctionnels :**
- [ ] UnifiedTTSManager opérationnel (4 handlers)
- [ ] Fallback automatique testé et validé
- [ ] Circuit breakers fonctionnels
- [ ] Cache LRU opérationnel
- [ ] Configuration YAML centralisée

### **✅ Critères Performance :**
- [ ] PiperNativeHandler <120ms (P95)
- [ ] Aucune régression vs baseline
- [ ] VRAM ≤10% RTX 3090
- [ ] Disponibilité 99.9%

### **✅ Critères Qualité :**
- [ ] Tests coverage >90%
- [ ] Type hints 100%
- [ ] Documentation complète
- [ ] Métriques Prometheus exportées

### **✅ Critères Validation Pratique :**
- [ ] **Tests réels exécutés** : test_tts_real.py, test_fallback_real.py, test_performance_real.py
- [ ] **Audio généré audible** : 4 fichiers test écoutés et validés
- [ ] **Qualité voix française** : Compréhensible et acceptable
- [ ] **Performance mesurée** : <120ms P95 pour piper_native confirmé
- [ ] **Fallback testé** : 4 niveaux validés avec audio généré

### **✅ Critères Déploiement :**
- [ ] Feature flags opérationnels
- [ ] Rollback script testé
- [ ] Archivage sécurisé
- [ ] Monitoring alertes configurées

---

## 📊 **MÉTRIQUES DE SUCCÈS**

### **🎯 KPIs Post-Déploiement :**

#### **Performance :**
- **Latence moyenne** : <120ms (vs <1000ms)
- **P95 latence** : <150ms
- **Cache hit rate** : >80%
- **Throughput** : >10 synthèses/s

#### **Robustesse :**
- **Uptime** : >99.9%
- **MTBF** : >168h
- **MTTR** : <5s
- **Fallback rate** : <1%

#### **Maintenance :**
- **Complexité code** : -87% fichiers
- **Time to fix** : -50%
- **Deployment time** : <5min
- **Rollback time** : <2min

---

## 🚀 **APPROBATION**

### **✅ Validation Stakeholders :**
- **Tech Lead :** Architecture approuvée
- **Product Owner :** Objectifs business validés
- **DevOps :** Infrastructure ready
- **QA :** Stratégie test approuvée

### **🎯 Go/No-Go Decision :**
**✅ GO pour implémentation Phase 2 Enterprise**

**Justification :**
- Architecture technique solide
- ROI élevé (-87% complexité, +88% performance)
- Risques maîtrisés avec mitigation
- Timeline réaliste (5.5 jours)

---

**🚀 Prêt pour implémentation UnifiedTTSManager enterprise-grade !** 