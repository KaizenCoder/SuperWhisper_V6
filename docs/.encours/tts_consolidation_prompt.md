# 🎯 PROMPT D'EXÉCUTION - CONSOLIDATION TTS SUPERWHISPER V6 (PHASE 2 ENTERPRISE)

**Date :** 2025-06-12  
**Version :** v2.0 Enterprise  
**Objectif :** Implémentation architecture UnifiedTTSManager enterprise-grade  

---

## 🚨 **MISSION CRITIQUE**

### **Objectif Principal :**
Implémenter l'architecture **UnifiedTTSManager enterprise-grade** en utilisant **EXCLUSIVEMENT le code expert fourni ci-dessous**, en remplaçant les 15 handlers fragmentés par une solution robuste <120ms.

### **🔥 INSTRUCTION ABSOLUE :**
**VOUS DEVEZ UTILISER LE CODE EXPERT FOURNI DANS CE PROMPT.** Ne pas réinventer ou modifier l'architecture. Le code a été validé par des experts et doit être implémenté tel quel.

### **Critères de Succès Impératifs :**

#### **1. Performance GPU Critique :**
- ✅ **PiperNativeHandler** fonctionnel sur RTX 3090 (CUDA:1)
- ✅ **Latence <120ms** pour synthèse principale (P95)
- ✅ **Allocation VRAM ≤10%** RTX 3090 (90% réservé LLM)

#### **2. Robustesse Enterprise :**
- ✅ **Fallback 4 niveaux** : Piper Native → Piper CLI → SAPI → Silent Emergency
- ✅ **Circuit Breakers** : Isolation automatique handlers défaillants
- ✅ **Cache LRU** : Phrases récurrentes <5ms
- ✅ **Monitoring Prometheus** : Métriques temps réel

#### **3. Architecture Clean :**
- ✅ **Configuration YAML** centralisée (config/tts.yaml)
- ✅ **Interface unifiée** : `async def synthesize() -> TTSResult`
- ✅ **13 handlers archivés** sécurisé avec rollback
- ✅ **Feature flags** : Déploiement progressif

---

## 📋 **CODE EXPERT À UTILISER OBLIGATOIREMENT**

### **🔧 Configuration YAML (config/tts.yaml) :**
```yaml
# config/tts.yaml
# Configuration unifiée du système TTS SuperWhisper V6

# ===================================================================
# SECTION PRINCIPALE
# ===================================================================
# Activation du handler Piper natif (performance optimale <120ms)
# Mettre à `false` pour forcer l'utilisation du fallback Piper CLI.
enable_piper_native: true

# ===================================================================
# CONFIGURATION DES BACKENDS
# ===================================================================
backends:
  # Priorité 1: Le plus rapide (GPU)
  piper_native:
    enabled: true
    model_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
    model_config_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx.json"
    device: "cuda:0"  # Pointera vers RTX 3090 après CUDA_VISIBLE_DEVICES
    speaker_id: 0
    target_latency_ms: 120

  # Priorité 2: Fallback fonctionnel (CPU)
  piper_cli:
    enabled: true
    model_path: "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
    executable_path: "piper/piper.exe" # Chemin vers l'exécutable
    speaker_id: 0
    target_latency_ms: 1000

  # Priorité 3: Fallback Windows natif (CPU)
  sapi_french:
    enabled: true
    voice_name: "Microsoft Hortense Desktop"
    rate: 0      # Vitesse de -10 (lent) à 10 (rapide)
    volume: 100  # Volume de 0 à 100
    target_latency_ms: 2000

  # Priorité 4: Ultime filet de sécurité
  silent_emergency:
    enabled: true
    log_level: "CRITICAL"
    alert_webhook: null # Optionnel: URL d'un webhook pour recevoir les alertes
    target_latency_ms: 5

# ===================================================================
# CONFIGURATION DES COMPOSANTS
# ===================================================================
# Cache pour les phrases récurrentes
cache:
  enabled: true
  max_size_mb: 100
  ttl_seconds: 3600 # 1 heure
  eviction_policy: "LRU" # Least Recently Used

# Disjoncteur pour isoler les backends défaillants
circuit_breaker:
  failure_threshold: 3 # Nombre d'échecs avant d'ouvrir le circuit
  reset_timeout_seconds: 30 # Temps avant de retenter un appel

# Monitoring via Prometheus
monitoring:
  prometheus_enabled: true
  prometheus_port: 9090
  log_performance_metrics: true
  alert_on_fallback: true # Log une alerte si un fallback est utilisé

# ===================================================================
# PARAMÈTRES AVANCÉS
# ===================================================================
advanced:
  # Fraction de VRAM allouée au processus TTS sur le GPU.
  # Laisser ~90% pour le LLM.
  gpu_memory_fraction: 0.1
  # Nombre de workers pour traiter les requêtes TTS en parallèle
  async_workers: 2
  # Limite de la longueur du texte pour éviter les abus
  max_text_length: 1000
  # Nettoyage automatique du texte (caractères non supportés, etc.)
  sanitize_text: true

# ===================================================================
# FEATURE FLAGS (Pour déploiement progressif)
# ===================================================================
feature_flags:
  # Flag principal pour activer le nouveau manager
  use_unified_tts: true
  # Flag pour forcer l'ancien mode (si un handler unique était utilisé)
  enable_legacy_mode: false
```

### **🏗️ Code Manager Principal (TTS/tts_manager.py) :**
```python
import asyncio
import hashlib
import time
import logging
import yaml
import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any
from pathlib import Path
import numpy as np

# Supposer que les librairies externes sont installées
# import torch
# from prometheus_client import Counter, Histogram, Gauge

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DATA CLASSES ET ENUMS ---
class TTSBackendType(Enum):
    PIPER_NATIVE = "piper_native"
    PIPER_CLI = "piper_cli"
    SAPI_FRENCH = "sapi_french"
    SILENT_EMERGENCY = "silent_emergency"
    CACHE = "cache"

@dataclass
class TTSResult:
    success: bool
    backend_used: str
    latency_ms: float
    audio_data: Optional[bytes] = None
    error: Optional[str] = None

# --- HANDLERS SPÉCIFIQUES ---
# NOTE: Ce sont des squelettes. L'implémentation réelle dépend des librairies.
class TTSHandler(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        pass

class PiperNativeHandler(TTSHandler):
    """Handler pour la lib Piper native (GPU)"""
    def __init__(self, config: dict):
        super().__init__(config)
        # Ex: self.voice = PiperVoice.load(config['model_path'])
        logging.info("Handler Piper Natif (GPU) initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        # NOTE: L'appel à la librairie est probablement bloquant
        # On l'exécute dans un thread pour ne pas bloquer l'event loop
        # audio_bytes = await asyncio.to_thread(self.voice.synthesize, text)
        # return audio_bytes
        await asyncio.sleep(0.1) # Simule la latence
        return b"fake_native_audio_data"

class PiperCliHandler(TTSHandler):
    """Handler pour Piper via ligne de commande (CPU)"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.executable_path = config['executable_path']
        self.model_path = config['model_path']
        logging.info("Handler Piper CLI (CPU) initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            self.executable_path,
            "--model", self.model_path,
            "--output_raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate(text.encode('utf-8'))
        if proc.returncode != 0:
            raise RuntimeError(f"Piper CLI a échoué: {stderr.decode()}")
        return stdout

class SapiFrenchHandler(TTSHandler):
    """Handler pour Windows SAPI"""
    def __init__(self, config: dict):
        super().__init__(config)
        # Ex: import win32com.client
        # self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
        logging.info("Handler SAPI Français initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        # Simule l'appel à SAPI et la récupération du flux audio
        await asyncio.sleep(1.5)
        return b"fake_sapi_audio_data"

class SilentEmergencyHandler(TTSHandler):
    """Handler d'urgence qui retourne un silence pour éviter un crash."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.log_level = config.get('log_level', 'CRITICAL')
        logging.info("Handler d'Urgence Silencieux initialisé.")

    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        logging.log(logging.getLevelName(self.log_level),
                    f"TTS EMERGENCY: Tous les backends ont échoué! Texte: '{text[:50]}...'")
        # Simuler l'envoi de webhook ici si configuré
        return self._generate_silent_wav()

    def _generate_silent_wav(self, duration_ms: int = 100) -> bytes:
        sample_rate = 22050
        num_samples = int(sample_rate * duration_ms / 1000)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
            wav.writeframes(b'\x00\x00' * num_samples)
        buffer.seek(0)
        return buffer.read()

# --- COMPOSANTS DE ROBUSTESSE ET PERFORMANCE ---
class CircuitBreaker:
    """Isole un service défaillant pour éviter de le surcharger."""
    def __init__(self, failure_threshold: int, reset_timeout: float):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"

    def is_open(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            logging.info("Circuit breaker est refermé.")

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            if self.state != "open":
                self.state = "open"
                self.last_failure_time = time.time()
                logging.warning(f"Circuit breaker est ouvert pour {self.reset_timeout}s.")

class TTSCache:
    """Cache en mémoire pour les synthèses fréquentes."""
    def __init__(self, config: dict):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = config.get('max_size_mb', 100) * 1024 * 1024
        self.ttl = config.get('ttl_seconds', 3600)
        self.current_size = 0

    def generate_key(self, text: str, config: Dict) -> str:
        key_str = f"{text}_{config.get('voice', 'default')}_{config.get('speed', 1.0)}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get(self, key: str) -> Optional[bytes]:
        entry = self.cache.get(key)
        if entry and (time.time() - entry['timestamp'] < self.ttl):
            return entry['audio_data']
        return None

    async def set(self, key: str, audio_data: bytes):
        size = len(audio_data)
        # NOTE: L'éviction LRU n'est pas implémentée ici pour la simplicité
        if self.current_size + size <= self.max_size:
            self.cache[key] = {'audio_data': audio_data, 'timestamp': time.time(), 'size': size}
            self.current_size += size

# --- LE MANAGER UNIFIÉ ---
class UnifiedTTSManager:
    """
    Gestionnaire unifié pour la synthèse vocale (Text-to-Speech).
    Orchestre plusieurs backends TTS avec fallback, cache, et monitoring.
    """
    def __init__(self, config: dict):
        self.config = config
        self._validate_gpu_configuration()

        # Initialisation des composants
        self.cache = TTSCache(config['cache'])
        cb_config = config['circuit_breaker']
        self.circuit_breakers = {
            backend: CircuitBreaker(cb_config['failure_threshold'], cb_config['reset_timeout_seconds'])
            for backend in TTSBackendType
        }
        self.handlers: Dict[TTSBackendType, TTSHandler] = {}
        self._initialize_handlers()
        logging.info("UnifiedTTSManager initialisé avec succès.")

    def _validate_gpu_configuration(self):
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                if "3090" not in device_name:
                    raise RuntimeError(f"GPU Invalide: {device_name}. RTX 3090 requise.")
                gpu_mem_fraction = self.config['advanced']['gpu_memory_fraction']
                torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, 0)
                logging.info(f"✅ RTX 3090 validée. Allocation mémoire GPU: {gpu_mem_fraction*100}%.")
            else:
                logging.warning("CUDA non disponible. Le backend piper_native sera désactivé.")
        except ImportError:
            logging.warning("PyTorch non trouvé. Le backend piper_native sera désactivé.")


    def _initialize_handlers(self):
        handler_map = {
            TTSBackendType.PIPER_NATIVE: PiperNativeHandler,
            TTSBackendType.PIPER_CLI: PiperCliHandler,
            TTSBackendType.SAPI_FRENCH: SapiFrenchHandler,
            TTSBackendType.SILENT_EMERGENCY: SilentEmergencyHandler
        }
        for backend_type, handler_class in handler_map.items():
            backend_name = backend_type.value
            if self.config['backends'].get(backend_name, {}).get('enabled', False):
                try:
                    if backend_type == TTSBackendType.PIPER_NATIVE and not self.config['enable_piper_native']:
                        continue
                    self.handlers[backend_type] = handler_class(self.config['backends'][backend_name])
                except Exception as e:
                    logging.error(f"Impossible d'initialiser le handler {backend_name}: {e}")

    async def synthesize(self, text: str, voice: Optional[str] = None,
                         speed: Optional[float] = None, reuse_cache: bool = True) -> TTSResult:
        # Docstring complet omis pour la concision (disponible dans la conversation précédente)
        start_time_total = time.perf_counter()
        
        # 1. Validation de l'input
        max_len = self.config['advanced']['max_text_length']
        if not text or len(text) > max_len:
            return TTSResult(success=False, backend_used="none", latency_ms=0, error=f"Texte invalide (vide ou > {max_len} chars).")

        # 2. Vérification du cache
        cache_key = self.cache.generate_key(text, {'voice': voice, 'speed': speed})
        if reuse_cache and (cached_audio := await self.cache.get(cache_key)):
            latency_ms = (time.perf_counter() - start_time_total) * 1000
            return TTSResult(success=True, backend_used=TTSBackendType.CACHE.value, latency_ms=latency_ms, audio_data=cached_audio)
        
        # 3. Chaîne de fallback
        # Créer une liste ordonnée des handlers activés
        backend_priority = {
            TTSBackendType.PIPER_NATIVE: 1,
            TTSBackendType.PIPER_CLI: 2,
            TTSBackendType.SAPI_FRENCH: 3,
            TTSBackendType.SILENT_EMERGENCY: 4
        }
        sorted_backends = sorted(self.handlers.keys(), key=lambda x: backend_priority[x])

        for backend_type in sorted_backends:
            if self.circuit_breakers[backend_type].is_open():
                continue

            try:
                start_time_handler = time.perf_counter()
                handler = self.handlers[backend_type]
                audio_data = await handler.synthesize(text, voice, speed)
                latency_ms = (time.perf_counter() - start_time_handler) * 1000

                self.circuit_breakers[backend_type].record_success()
                await self.cache.set(cache_key, audio_data)

                # Validation performance
                target_latency = self.config['backends'][backend_type.value]['target_latency_ms']
                if latency_ms > target_latency:
                    logging.warning(f"Performance Warning: {backend_type.value} a dépassé sa cible de latence ({latency_ms:.0f}ms > {target_latency}ms).")

                return TTSResult(success=True, backend_used=backend_type.value, latency_ms=latency_ms, audio_data=audio_data)

            except Exception as e:
                logging.error(f"Échec du backend {backend_type.value}: {e}")
                self.circuit_breakers[backend_type].record_failure()
                continue
        
        # Si tous les backends ont échoué
        return TTSResult(success=False, backend_used="none", latency_ms=0, error="Tous les backends TTS ont échoué, y compris l'handler d'urgence.")
```

---

## 🔧 **CONTRAINTES TECHNIQUES ABSOLUES**

### **🚨 RÈGLE ABSOLUE - Stockage Modèles :**
**TOUS LES MODÈLES TTS/LLM DOIVENT ÊTRE STOCKÉS EXCLUSIVEMENT SUR LE DISQUE D:**
- **Répertoire obligatoire :** `D:\TTS_Voices\`
- **Modèles disponibles :** 
  ```
  D:\TTS_Voices\piper\fr_FR-siwis-medium.onnx (63MB) ✅ DISPONIBLE
  D:\TTS_Voices\piper\fr_FR-siwis-medium.onnx.json ✅ DISPONIBLE
  D:\TTS_Voices\piper\fr_FR-mls_1840-medium.onnx ✅ DISPONIBLE  
  D:\TTS_Voices\piper\fr_FR-upmc-medium.onnx ✅ DISPONIBLE
  ```
- **INTERDICTION ABSOLUE :** Aucun stockage de modèle ailleurs que sur D:\
- **AVANT TÉLÉCHARGEMENT :** Vérifier disponibilité dans `D:\TTS_Voices\`

### **🚨 Configuration GPU Obligatoire :**
Le code expert intègre déjà la validation GPU RTX 3090 dans `_validate_gpu_configuration()`. **UTILISEZ CETTE MÉTHODE TELLE QUELLE.**

### **🎯 Architecture Handlers Obligatoire :**
Les 4 handlers sont **DÉJÀ DÉFINIS** dans le code expert :
- `PiperNativeHandler` (GPU <120ms)
- `PiperCliHandler` (CPU <1000ms) 
- `SapiFrenchHandler` (SAPI <2000ms)
- `SilentEmergencyHandler` (Silence <5ms)

### **🔒 Interface API Standardisée :**
L'interface `TTSResult` et la méthode `synthesize()` sont **DÉJÀ IMPLÉMENTÉES** dans le code expert.

---

## 📋 **SPÉCIFICATIONS IMPLÉMENTATION**

### **🔥 RÈGLE ABSOLUE :**
**IMPLÉMENTEZ LE CODE EXPERT TEL QUEL.** Voici les adaptations nécessaires :

#### **1. Réparation PiperNativeHandler (PRIORITÉ 1) :**
```python
# Dans PiperNativeHandler.synthesize(), remplacez la simulation par :
# 1. Installer piper-python : pip install piper-tts
# 2. Remplacer les commentaires par l'implémentation réelle :
#    from piper import PiperVoice
#    self.voice = PiperVoice.load(config['model_path'])
#    audio_bytes = await asyncio.to_thread(self.voice.synthesize, text)
# 3. Valider latence <120ms
```

#### **2. Adaptation PiperCliHandler :**
```python
# Le code expert est CORRECT. Adaptez uniquement les chemins :
# - executable_path: "piper/piper.exe" → chemin réel
# - model_path: DÉJÀ CORRECT → "D:/TTS_Voices/piper/fr_FR-siwis-medium.onnx"
```

#### **3. Adaptation SapiFrenchHandler :**
```python
# Remplacez la simulation par l'implémentation SAPI réelle :
# import win32com.client
# self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
# Utilisez le handler SAPI existant comme référence
```

#### **4. Configuration YAML :**
**CRÉEZ EXACTEMENT** le fichier `config/tts.yaml` avec le contenu expert fourni.

#### **5. Tests Validation :**
```python
# Créez tests/test_unified_tts_manager.py avec :
# - Test fallback automatique (simulation pannes)
# - Test circuit breaker (3 échecs → isolation)
# - Test cache (hit <5ms)
# - Test performance (<120ms PiperNative)
```

---

## 🎖️ **LIVRABLES ATTENDUS**

### **📁 Fichiers à Créer :**
```
config/tts.yaml                    # Configuration centralisée
TTS/tts_manager_unified.py         # Manager principal
TTS/handlers/piper_native.py       # Handler GPU réparé
TTS/handlers/piper_cli.py          # Handler CLI (existant adapté)
TTS/handlers/sapi_french.py        # Handler SAPI (existant adapté)
TTS/handlers/silent_emergency.py   # Handler urgence
TTS/components/circuit_breaker.py  # Circuit breaker
TTS/components/cache.py            # Cache LRU
tests/test_unified_tts_manager.py  # Tests complets
```

### **📁 Fichiers à Archiver :**
```
TTS/legacy_handlers_20250612/      # 13 handlers obsolètes
├── README_ROLLBACK.md             # Instructions restauration
├── tts_handler_piper_*.py (10)    # Handlers Piper redondants
├── tts_handler_coqui.py           # Handler Coqui
├── tts_handler_mvp.py             # Handler MVP
└── tts_handler_fallback.py        # Handler fallback basique
```

### **📁 Fichiers à Modifier :**
```
run_assistant.py                   # Intégration UnifiedTTSManager
requirements.txt                   # Dépendances (piper-python, prometheus)
```

---

## 🚨 **POINTS DE VALIDATION OBLIGATOIRES**

### **🎯 Checkpoints Bloquants :**

#### **Checkpoint 1 - PiperNativeHandler :**
- ✅ Handler GPU fonctionnel sans erreur
- ✅ Latence <120ms validée (3 tests minimum)
- ✅ VRAM ≤10% RTX 3090 confirmée
- ✅ **TEST RÉEL OBLIGATOIRE :** `python test_tts_real.py` → Écouter audio généré
- ✅ **VALIDATION MANUELLE :** Qualité voix française acceptable
- ❌ **STOP si échec** → Fallback architecture actuelle

#### **Checkpoint 2 - UnifiedTTSManager :**
- ✅ 4 handlers intégrés et fonctionnels
- ✅ Fallback automatique testé (simulation pannes)
- ✅ Configuration YAML opérationnelle
- ✅ **TEST FALLBACK RÉEL :** `python test_fallback_real.py` → 4 niveaux validés
- ✅ **ÉCOUTE COMPARATIVE :** Audio de chaque backend (piper_native vs piper_cli vs sapi)
- ✅ Tests unitaires 100% passants

#### **Checkpoint 3 - Déploiement :**
- ✅ Feature flag activation réussie
- ✅ Métriques Prometheus fonctionnelles
- ✅ Performance ≥ baseline (pas de régression)
- ✅ Archivage sécurisé + rollback testé

---

## 🎖️ **CRITÈRES D'ACCEPTATION FINALE**

### **✅ Performance :**
- Latence PiperNative <120ms (P95)
- Latence PiperCLI <1000ms
- Latence SAPI <2000ms
- Cache hit <5ms

### **✅ Robustesse :**
- Disponibilité 99.9% (fallback)
- Circuit breakers fonctionnels
- Recovery automatique
- Monitoring temps réel

### **✅ Qualité Code :**
- Type hints 100%
- Docstrings complètes
- Tests coverage >90%
- Configuration externalisée

### **✅ Déploiement :**
- Feature flags opérationnels
- Rollback script testé
- Documentation complète
- Métriques exportées

---

## 🚨 **INSTRUCTIONS D'UTILISATION DU CODE EXPERT**

### **📋 Étapes d'Implémentation Obligatoires :**

#### **Étape 1 - Vérification Modèles & Création Fichiers :**
```bash
# 1. OBLIGATOIRE - Vérifier modèles disponibles sur D:\
Get-ChildItem "D:\TTS_Voices\piper" -Name
# ✅ Confirmer présence : fr_FR-siwis-medium.onnx (63MB) + .json

# 2. Créer config/tts.yaml avec le YAML expert fourni (chemins D:\ déjà corrects)
mkdir -p config
# Copier le contenu YAML expert dans config/tts.yaml

# 3. Créer TTS/tts_manager.py avec le code Python expert fourni  
# Copier le code Python expert dans TTS/tts_manager.py

# 4. Archiver les 13 handlers obsolètes
mkdir -p TTS/legacy_handlers_20250612
mv TTS/tts_handler_piper_*.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_coqui.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_mvp.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_fallback.py TTS/legacy_handlers_20250612/
```

#### **Étape 2 - Adapter les handlers simulés :**
```python
# Dans PiperNativeHandler.synthesize() :
# Remplacer : await asyncio.sleep(0.1); return b"fake_native_audio_data"
# Par : Implémentation piper-python réelle

# Dans SapiFrenchHandler.synthesize() :
# Remplacer : await asyncio.sleep(1.5); return b"fake_sapi_audio_data"  
# Par : Implémentation SAPI réelle (référence handler existant)
```

#### **Étape 3 - Intégration dans run_assistant.py :**
```python
# Remplacer l'ancien handler TTS par :
import yaml
from TTS.tts_manager import UnifiedTTSManager

# Chargement configuration
with open('config/tts.yaml', 'r') as f:
    tts_config = yaml.safe_load(f)

# Initialisation manager unifié
tts_manager = UnifiedTTSManager(tts_config)

# Utilisation
async def process_tts(text: str):
    result = await tts_manager.synthesize(text)
    if result.success:
        # Traitement audio result.audio_data
        print(f"✅ TTS: {result.backend_used} ({result.latency_ms:.0f}ms)")
    else:
        print(f"❌ TTS échec: {result.error}")
```

#### **Étape 4 - Tests Réels Pratiques :**
```python
# 1. SCRIPT DE TEST RÉEL - Créer test_tts_real.py
"""
Script de test pratique pour validation manuelle pendant l'implémentation.
Génère des fichiers audio réels pour écoute et validation.
"""
import asyncio
import time
import yaml
from pathlib import Path
from TTS.tts_manager import UnifiedTTSManager

async def test_real_tts():
    # Chargement config
    with open('config/tts.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    manager = UnifiedTTSManager(config)
    
    # Tests réels avec phrases françaises
    test_phrases = [
        "Bonjour, je suis votre assistant vocal SuperWhisper.",
        "La synthèse vocale fonctionne parfaitement avec RTX 3090.",
        "Test de performance et de qualité audio en français.",
        "Validation du fallback automatique en cas d'erreur."
    ]
    
    print("🎤 TESTS TTS RÉELS - Génération fichiers audio")
    print("=" * 60)
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n📝 Test {i}/4: '{phrase[:30]}...'")
        
        start_time = time.perf_counter()
        result = await manager.synthesize(phrase)
        latency = (time.perf_counter() - start_time) * 1000
        
        if result.success:
            # Sauvegarder audio pour écoute
            audio_file = f"test_output/test_{i}_{result.backend_used}.wav"
            Path("test_output").mkdir(exist_ok=True)
            
            with open(audio_file, 'wb') as f:
                f.write(result.audio_data)
            
            print(f"✅ Backend: {result.backend_used}")
            print(f"✅ Latence: {result.latency_ms:.0f}ms (mesurée: {latency:.0f}ms)")
            print(f"✅ Audio: {audio_file} ({len(result.audio_data)} bytes)")
            print(f"🎧 ÉCOUTER: start {audio_file}")
        else:
            print(f"❌ ÉCHEC: {result.error}")
    
    print(f"\n🎯 VALIDATION MANUELLE:")
    print(f"1. Écouter les 4 fichiers dans test_output/")
    print(f"2. Vérifier qualité audio française")
    print(f"3. Confirmer latence <120ms pour piper_native")
    print(f"4. Tester fallback en désactivant handlers")

if __name__ == "__main__":
    asyncio.run(test_real_tts())
```

```python
# 2. SCRIPT TEST FALLBACK RÉEL - Créer test_fallback_real.py
"""
Test pratique du système de fallback avec simulation de pannes.
"""
import asyncio
import yaml
from TTS.tts_manager import UnifiedTTSManager

async def test_fallback_simulation():
    print("🔧 TEST FALLBACK RÉEL - Simulation pannes")
    
    with open('config/tts.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test 1: Tous handlers actifs
    print("\n1️⃣ Test normal (tous handlers actifs)")
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test normal avec tous les backends.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 2: Désactiver piper_native (forcer fallback)
    print("\n2️⃣ Test fallback (piper_native désactivé)")
    config['backends']['piper_native']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test fallback vers piper CLI.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 3: Désactiver piper_native + piper_cli (forcer SAPI)
    print("\n3️⃣ Test fallback SAPI (piper désactivés)")
    config['backends']['piper_cli']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test fallback vers SAPI français.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    # Test 4: Tous désactivés sauf emergency
    print("\n4️⃣ Test emergency (tous backends désactivés)")
    config['backends']['sapi_french']['enabled'] = False
    manager = UnifiedTTSManager(config)
    result = await manager.synthesize("Test handler d'urgence silencieux.")
    print(f"✅ Backend utilisé: {result.backend_used} ({result.latency_ms:.0f}ms)")
    
    print(f"\n🎯 VALIDATION: Chaîne de fallback complète testée!")

if __name__ == "__main__":
    asyncio.run(test_fallback_simulation())
```

```bash
# 3. COMMANDES TEST RAPIDES
# Test GPU RTX 3090
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CUDA indisponible')"

# Test modèles disponibles
Get-ChildItem "D:\TTS_Voices\piper" -Name

# Test manager (après implémentation)
python test_tts_real.py

# Test fallback
python test_fallback_real.py

# Écoute rapide des résultats
start test_output\test_1_piper_native.wav
start test_output\test_2_piper_cli.wav
```

```python
# 4. SCRIPT BENCHMARK PERFORMANCE RÉEL - Créer test_performance_real.py
"""
Benchmark de performance avec mesures réelles et validation des KPI.
"""
import asyncio
import time
import statistics
import yaml
from TTS.tts_manager import UnifiedTTSManager

async def benchmark_performance():
    print("⚡ BENCHMARK PERFORMANCE RÉEL")
    print("=" * 50)
    
    with open('config/tts.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    manager = UnifiedTTSManager(config)
    
    # Phrases de test de différentes longueurs
    test_cases = [
        ("Court", "Bonjour."),
        ("Moyen", "Bonjour, je suis votre assistant vocal SuperWhisper."),
        ("Long", "Bonjour, je suis votre assistant vocal SuperWhisper. La synthèse vocale fonctionne parfaitement avec la carte graphique RTX 3090 et les modèles Piper français."),
    ]
    
    results = {}
    
    for case_name, text in test_cases:
        print(f"\n📊 Test {case_name}: '{text[:40]}...'")
        latencies = []
        
        # 10 mesures pour statistiques fiables
        for i in range(10):
            start_time = time.perf_counter()
            result = await manager.synthesize(text)
            latency = (time.perf_counter() - start_time) * 1000
            
            if result.success:
                latencies.append(latency)
                print(f"  Run {i+1:2d}: {latency:6.1f}ms ({result.backend_used})")
            else:
                print(f"  Run {i+1:2d}: ÉCHEC - {result.error}")
        
        if latencies:
            results[case_name] = {
                'mean': statistics.mean(latencies),
                'p95': sorted(latencies)[int(0.95 * len(latencies))],
                'min': min(latencies),
                'max': max(latencies),
                'backend': result.backend_used
            }
    
    # Rapport final
    print(f"\n🎯 RAPPORT PERFORMANCE FINALE")
    print("=" * 50)
    for case_name, stats in results.items():
        print(f"{case_name:6s}: Moy={stats['mean']:6.1f}ms | P95={stats['p95']:6.1f}ms | Min={stats['min']:6.1f}ms | Max={stats['max']:6.1f}ms | Backend={stats['backend']}")
    
    # Validation KPI
    print(f"\n✅ VALIDATION KPI:")
    if 'Court' in results:
        p95_court = results['Court']['p95']
        backend = results['Court']['backend']
        if backend == 'piper_native' and p95_court < 120:
            print(f"✅ PiperNative P95: {p95_court:.1f}ms < 120ms TARGET")
        elif backend == 'piper_cli' and p95_court < 1000:
            print(f"✅ PiperCLI P95: {p95_court:.1f}ms < 1000ms TARGET")
        else:
            print(f"❌ Performance insuffisante: {backend} P95={p95_court:.1f}ms")

if __name__ == "__main__":
    asyncio.run(benchmark_performance())
```

```bash
# 5. VALIDATION COMPLÈTE PENDANT IMPLÉMENTATION
echo "🧪 TESTS RÉELS SUPERWHISPER TTS"

# Étape 1: Vérification environnement
python -c "import torch; print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CUDA indisponible')"

# Étape 2: Test fonctionnel de base
python test_tts_real.py

# Étape 3: Test robustesse fallback  
python test_fallback_real.py

# Étape 4: Benchmark performance
python test_performance_real.py

# Étape 5: Écoute validation manuelle
echo "🎧 VALIDATION MANUELLE - Écouter les fichiers:"
dir test_output\*.wav
echo "Commande: start test_output\test_1_piper_native.wav"
```

---

## 🚀 **COMMANDES DE DÉMARRAGE**

```bash
# Initialisation projet
git checkout -b feature/tts-enterprise-consolidation
git tag pre-tts-enterprise-consolidation

# Validation environnement GPU RTX 3090
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Installation dépendances
pip install piper-tts pyyaml

# Test du manager unifié
python -c "
import yaml
from TTS.tts_manager import UnifiedTTSManager
with open('config/tts.yaml', 'r') as f:
    config = yaml.safe_load(f)
manager = UnifiedTTSManager(config)
print('✅ UnifiedTTSManager initialisé avec succès')
"

# Démarrage implémentation
echo "🚀 Phase 2 Enterprise - Consolidation TTS avec code expert démarrée"
```

---

## 🎯 **VALIDATION FINALE**

### **Critères d'Acceptation :**
- [ ] Code expert implémenté à 100% sans modification
- [ ] Configuration YAML fonctionnelle
- [ ] 4 handlers intégrés (PiperNative, PiperCLI, SAPI, SilentEmergency)
- [ ] Fallback automatique testé
- [ ] Circuit breakers opérationnels
- [ ] Cache LRU fonctionnel
- [ ] Performance <120ms validée (PiperNative)
- [ ] 13 handlers obsolètes archivés
- [ ] Tests unitaires + intégration passants

### **🎧 VALIDATION PRATIQUE OBLIGATOIRE :**

#### **Tests Réels à Effectuer Pendant l'Implémentation :**
1. **`python test_tts_real.py`** → Génère 4 fichiers audio à écouter
2. **`python test_fallback_real.py`** → Valide la chaîne de fallback complète  
3. **`python test_performance_real.py`** → Benchmark avec 10 mesures par cas
4. **Écoute manuelle** → Validation qualité voix française
5. **Test intégration** → Remplacement dans `run_assistant.py`

#### **Critères d'Acceptation Pratiques :**
- ✅ **Audio généré audible** et compréhensible en français
- ✅ **Latence mesurée** <120ms pour piper_native (P95)
- ✅ **Fallback fonctionnel** vers piper_cli puis SAPI puis emergency
- ✅ **Qualité vocale** acceptable pour assistant vocal
- ✅ **Intégration réussie** dans SuperWhisper sans régression

#### **🚨 Points de Blocage :**
- ❌ **Audio inaudible/corrompu** → STOP implémentation
- ❌ **Latence >200ms** sur piper_native → Optimisation GPU requise
- ❌ **Fallback non fonctionnel** → Architecture à revoir
- ❌ **Crash/erreur** sur phrases françaises → Debug handlers

### **🔥 RAPPEL CRITIQUE :**
**UTILISEZ EXCLUSIVEMENT LE CODE EXPERT FOURNI.** Toute modification de l'architecture annule la garantie de performance et de robustesse validée par les experts.

**VALIDATION OBLIGATOIRE :** Tests réels avec écoute manuelle avant déploiement.

**GO ! Implémentation architecture enterprise UnifiedTTSManager avec code expert validé.** 