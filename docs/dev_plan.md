# 📋 PLAN DE DÉVELOPPEMENT - PHASE 4 STT SUPERWHISPER V6

**Version :** 4.1 VALIDATIONS HUMAINES  
**Date :** 12 juin 2025  
**Configuration :** RTX 3090 Unique (24GB VRAM)  
**Durée estimée :** 3 jours + validations humaines  
**Objectif :** Pipeline voix-à-voix avec validation humaine continue  

---

## 🎯 CONTEXTE ET OBJECTIFS

### **État Actuel SuperWhisper V6**
- ✅ **Phase 3 TTS** : Terminée avec succès exceptionnel (29.5ms latence cache)
- ✅ **Configuration GPU** : RTX 3090 exclusive validée (24GB VRAM)
- ✅ **Architecture** : UnifiedTTSManager opérationnel avec 4 backends
- ✅ **Performance** : Dépasse tous objectifs (+340% amélioration)

### **Mission Phase 4**
Intégrer module STT avec **Prism_Whisper2** sur RTX 3090 unique pour compléter le pipeline **voix-à-voix** (STT → LLM → TTS) avec performance < 730ms.

### **Configuration GPU Réelle**
```
Hardware Dual-GPU :
├── Bus PCI 0 : RTX 5060 Ti (16GB) ❌ INTERDITE
└── Bus PCI 1 : RTX 3090 (24GB) ✅ SEULE AUTORISÉE

Mapping Software :
CUDA_VISIBLE_DEVICES='1' → cuda:0 = RTX 3090 (24GB)
```

---

## 🚨 PRÉREQUIS CRITIQUES - VALIDATION HUMAINE

### **📋 Protocole Validation Audio Obligatoire**
**RÈGLE ABSOLUE** : Les tests audio au microphone DOIVENT inclure une validation humaine par écoute manuelle.

#### **🔄 Cycle de Développement Audio**
```
1. ⚡ IMPLÉMENTATION STT
   ↓
2. 🧪 TESTS AUTOMATISÉS (métriques, performance)
   ↓
3. 🎧 TESTS AUDIO MICROPHONE
   ↓
4. 👂 VALIDATION HUMAINE AUDIO (OBLIGATOIRE)
   ↓
5. 📝 DOCUMENTATION (journal + suivi)
   ↓
6. ✅ PASSAGE ÉTAPE SUIVANTE
```

#### **📚 Documentation Continue Obligatoire**

##### **📝 Journal de Développement**
- **Fichier :** `docs/journal_developpement.md`
- **Règle :** ❌ INTERDICTION suppression, ✅ MODIFICATION uniquement
- **Mise à jour :** **OBLIGATOIRE** avant chaque commit
- **Contenu :** Décisions, problèmes, solutions, validations

##### **📊 Suivi Phase 4 STT**
- **Fichier :** `docs/suivi_stt_phase4.md` (créer automatiquement jour 1)
- **Template :** Basé sur `docs/suivi_consolidation_tts_phase2.md`
- **Mise à jour :** Temps réel avec chaque avancement
- **Contenu :** Progression, validations humaines, métriques

#### **⚠️ Responsabilités Développeur**
- **AVANT** chaque étape : Mise à jour documentation
- **PENDANT** chaque étape : Tests avec validation humaine planifiée
- **APRÈS** chaque étape : Documentation résultats et décisions

---

## 📅 PLANNING DÉTAILLÉ - 3 JOURS

### **🚀 JOUR 1 - POC PRISM_WHISPER2 RTX 3090**

#### **📝 DÉBUT JOUR 1 - Documentation Obligatoire (30 min)**
```bash
# ÉTAPE 0.1 - Création Suivi Phase 4 STT (OBLIGATOIRE)
cp docs/suivi_consolidation_tts_phase2.md docs/suivi_stt_phase4.md
# Adapter template pour Phase 4 STT
# Documenter objectifs jour 1

# ÉTAPE 0.2 - Mise à jour Journal Développement (OBLIGATOIRE)  
# Ajouter entrée session développement jour 1
# Documenter plan et objectifs
```

#### **Matin (3.5h) - Setup et Validation**
```bash
# ÉTAPE 1.1 - Validation Configuration RTX 3090 (30 min)
python test_gpu_correct.py
python test_validation_rtx3090_detection.py
nvidia-smi  # Confirmer RTX 3090 disponible

# ÉTAPE 1.2 - Installation Prism_Whisper2 (30 min)
pip install prism-whisper2
pip install prometheus-client
pip install asyncio-throttle

# ÉTAPE 1.3 - Validation Dual-GPU (30 min)
python scripts/validate_dual_gpu_rtx3090.py
```

#### **ÉTAPE 1.4 - Backend PrismSTTBackend (2.5h)**
**Créer :** `STT/backends/prism_stt_backend.py`

```python
#!/usr/bin/env python3
"""
Backend Prism STT SuperWhisper V6 - RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (cuda:0) OBLIGATOIRE
"""

import os
# Configuration GPU RTX 3090 - OBLIGATOIRE AVANT IMPORT
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

import torch
import time
import asyncio
import numpy as np
from typing import Dict, Any
import prism_whisper2

def validate_rtx3090_mandatory():
    """Validation RTX 3090 selon standards SuperWhisper V6"""
    # [Code validation complète selon standards]
    pass

class PrismSTTBackend:
    def __init__(self, model_size: str = "large-v2"):
        validate_rtx3090_mandatory()
        
        self.device = "cuda:0"  # RTX 3090 après mapping
        self.model_size = model_size
        
        # Charger modèle en float16 sur RTX 3090
        self.model = prism_whisper2.load_model(
            model_size,
            device=self.device,
            compute_type="float16"
        )
        
        # Métriques
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0
        
    async def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcription asynchrone RTF < 1.0"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Transcription sur RTX 3090
            result = await self.model.transcribe_async(audio)
            
            # Métriques performance
            duration = time.time() - start_time
            audio_duration = len(audio) / 16000
            rtf = duration / audio_duration
            
            self.total_latency += duration
            
            return {
                "text": result.text,
                "confidence": result.confidence,
                "latency_ms": duration * 1000,
                "rtf": rtf,
                "model_size": self.model_size
            }
            
        except Exception as e:
            self.total_errors += 1
            raise Exception(f"Erreur Prism STT: {e}")
```

#### **Après-midi (3.5h) - Tests PoC + Validation Audio**

#### **ÉTAPE 1.5 - Tests PoC (1.5h)**
**Créer :** `tests/test_prism_poc.py`

```python
#!/usr/bin/env python3
"""Tests PoC Prism STT - RTX 3090"""

import os
# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import pytest
import numpy as np
import asyncio
from STT.backends.prism_stt_backend import PrismSTTBackend

def generate_test_audio(duration: float = 5.0, sample_rate: int = 16000):
    """Génère audio test (silence + bip)"""
    samples = int(duration * sample_rate)
    audio = np.random.normal(0, 0.1, samples).astype(np.float32)
    
    # Ajouter bip à 1kHz
    t = np.linspace(0, duration, samples)
    beep = 0.3 * np.sin(2 * np.pi * 1000 * t)
    audio[samples//2:samples//2+sample_rate//4] += beep[:sample_rate//4]
    
    return audio

@pytest.mark.asyncio
async def test_prism_large_rtx3090():
    """Test PoC Prism large-v2 sur RTX 3090"""
    
    # Backend large-v2
    backend = PrismSTTBackend(model_size="large-v2")
    
    # Audio test 5 secondes
    audio = generate_test_audio(duration=5.0)
    
    # Transcription
    result = await backend.transcribe(audio)
    
    # Validations RTX 3090
    assert result["rtf"] < 1.0, f"RTF {result['rtf']:.2f} > 1.0"
    assert result["latency_ms"] < 400, f"Latence {result['latency_ms']:.0f}ms > 400ms"
    assert result["confidence"] > 0.3, f"Confiance {result['confidence']:.2f} trop faible"
    
    print(f"✅ Prism Large RTX 3090:")
    print(f"   RTF: {result['rtf']:.2f}")
    print(f"   Latence: {result['latency_ms']:.0f}ms")
    print(f"   Confiance: {result['confidence']:.2f}")

@pytest.mark.asyncio
async def test_prism_tiny_rtx3090():
    """Test PoC Prism tiny sur RTX 3090"""
    
    backend = PrismSTTBackend(model_size="tiny")
    audio = generate_test_audio(duration=3.0)
    
    result = await backend.transcribe(audio)
    
    # Tiny doit être plus rapide
    assert result["rtf"] < 0.5, f"RTF tiny {result['rtf']:.2f} > 0.5"
    assert result["latency_ms"] < 200, f"Latence tiny {result['latency_ms']:.0f}ms > 200ms"
    
    print(f"✅ Prism Tiny RTX 3090:")
    print(f"   RTF: {result['rtf']:.2f}")
    print(f"   Latence: {result['latency_ms']:.0f}ms")

if __name__ == "__main__":
    asyncio.run(test_prism_large_rtx3090())
    asyncio.run(test_prism_tiny_rtx3090())
```

#### **ÉTAPE 1.6 - 🎧 VALIDATION HUMAINE AUDIO OBLIGATOIRE (1h)**
**RÈGLE ABSOLUE** : Tests microphone avec écoute manuelle obligatoire

**Créer :** `scripts/demo_stt_microphone_validation.py`

```python
#!/usr/bin/env python3
"""
Démonstration STT avec microphone - VALIDATION HUMAINE OBLIGATOIRE
🚨 CONFIGURATION GPU: RTX 3090 (cuda:0) OBLIGATOIRE
"""

import os
# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import pyaudio
import numpy as np
import time
import asyncio
import json
from STT.backends.prism_stt_backend import PrismSTTBackend

async def validation_humaine_audio_microphone():
    """
    🎧 VALIDATION HUMAINE AUDIO OBLIGATOIRE
    Tests microphone avec écoute manuelle par utilisateur humain
    """
    
    print("🎧 VALIDATION HUMAINE AUDIO - TESTS MICROPHONE")
    print("=" * 60)
    print("🚨 RÈGLE ABSOLUE : Validation par écoute humaine obligatoire")
    print()
    
    # Initialiser backend STT
    backend = PrismSTTBackend(model_size="large-v2")
    
    # Configuration audio
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    
    audio = pyaudio.PyAudio()
    
    tests_validation = [
        {
            "nom": "Test Phrase Courte",
            "duree": 3,
            "instruction": "Dites une phrase courte (ex: 'Bonjour comment allez-vous')"
        },
        {
            "nom": "Test Phrase Longue", 
            "duree": 8,
            "instruction": "Dites une phrase longue avec détails"
        },
        {
            "nom": "Test Conditions Variables",
            "duree": 5,
            "instruction": "Parlez avec bruit ambiant ou distance variable"
        }
    ]
    
    resultats_validation = []
    
    for i, test in enumerate(tests_validation, 1):
        print(f"\n🎯 TEST {i}/3 : {test['nom']}")
        print(f"📝 Instruction : {test['instruction']}")
        print(f"⏱️ Durée : {test['duree']} secondes")
        
        input("🎤 Appuyez sur Entrée quand vous êtes prêt à parler...")
        
        # Enregistrement
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print(f"🔴 ENREGISTREMENT... ({test['duree']}s)")
        
        frames = []
        for _ in range(0, int(RATE / CHUNK * test['duree'])):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.float32))
        
        stream.stop_stream()
        stream.close()
        
        # Transcription
        audio_data = np.concatenate(frames)
        start_time = time.time()
        
        try:
            result = await backend.transcribe(audio_data)
            latence = time.time() - start_time
            
            print(f"✅ Transcription terminée en {latence*1000:.0f}ms")
            print(f"📝 Texte transcrit : '{result['text']}'")
            print(f"📊 Confiance : {result['confidence']:.2f}")
            print(f"⚡ RTF : {result['rtf']:.2f}")
            
            # 🎧 VALIDATION HUMAINE OBLIGATOIRE
            print("\n" + "="*50)
            print("🎧 VALIDATION HUMAINE AUDIO OBLIGATOIRE")
            print("="*50)
            
            print("👂 Écoutez attentivement et évaluez :")
            print(f"   Texte transcrit : '{result['text']}'")
            print()
            
            # Validation par utilisateur humain
            while True:
                precision = input("🎯 Précision transcription (excellent/bon/acceptable/insuffisant) : ").lower()
                if precision in ['excellent', 'bon', 'acceptable', 'insuffisant']:
                    break
                print("❌ Réponse invalide. Utilisez : excellent/bon/acceptable/insuffisant")
            
            while True:
                latence_percue = input("⏱️ Latence perçue (imperceptible/acceptable/gênante) : ").lower()
                if latence_percue in ['imperceptible', 'acceptable', 'gênante']:
                    break
                print("❌ Réponse invalide. Utilisez : imperceptible/acceptable/gênante")
            
            commentaires = input("💬 Commentaires détaillés (optionnel) : ")
            
            # Validation finale
            while True:
                validation = input("🎯 Validation finale (validé/à_corriger/validé_avec_réserves) : ").lower()
                if validation in ['validé', 'à_corriger', 'validé_avec_réserves']:
                    break
                print("❌ Réponse invalide. Utilisez : validé/à_corriger/validé_avec_réserves")
            
            # Enregistrer résultat validation
            resultats_validation.append({
                "test": test['nom'],
                "texte_transcrit": result['text'],
                "latence_ms": latence * 1000,
                "rtf": result['rtf'],
                "confiance": result['confidence'],
                "precision_humaine": precision,
                "latence_percue": latence_percue,
                "commentaires": commentaires,
                "validation_finale": validation,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"✅ Validation {validation.upper()} enregistrée")
            
        except Exception as e:
            print(f"❌ Erreur transcription : {e}")
            resultats_validation.append({
                "test": test['nom'],
                "erreur": str(e),
                "validation_finale": "échec",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    audio.terminate()
    
    # Rapport final validation humaine
    print("\n" + "="*60)
    print("📊 RAPPORT VALIDATION HUMAINE AUDIO - JOUR 1")
    print("="*60)
    
    for result in resultats_validation:
        print(f"\n🎯 {result['test']} :")
        if 'erreur' not in result:
            print(f"   📝 Transcription : '{result['texte_transcrit']}'")
            print(f"   ⏱️ Latence : {result['latence_ms']:.0f}ms")
            print(f"   🎯 Précision humaine : {result['precision_humaine']}")
            print(f"   ⏱️ Latence perçue : {result['latence_percue']}")
            print(f"   ✅ Validation : {result['validation_finale']}")
            if result['commentaires']:
                print(f"   💬 Commentaires : {result['commentaires']}")
        else:
            print(f"   ❌ Erreur : {result['erreur']}")
    
    # Sauvegarder rapport
    os.makedirs('test_output', exist_ok=True)
    with open('test_output/validation_humaine_jour1.json', 'w', encoding='utf-8') as f:
        json.dump(resultats_validation, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Rapport sauvegardé : test_output/validation_humaine_jour1.json")
    print("\n🎊 VALIDATION HUMAINE AUDIO JOUR 1 TERMINÉE")

if __name__ == "__main__":
    asyncio.run(validation_humaine_audio_microphone())
```

#### **ÉTAPE 1.7 - Documentation Obligatoire (30 min)**
```bash
# Mise à jour journal développement (OBLIGATOIRE)
# Ajouter entrée session jour 1 avec :
# - Objectifs atteints
# - Résultats validation humaine audio
# - Décisions techniques prises
# - Problèmes rencontrés et solutions

# Mise à jour suivi STT Phase 4 (OBLIGATOIRE)
# Mettre à jour docs/suivi_stt_phase4.md avec :
# - Progression jour 1
# - Résultats validations humaines
# - Métriques performance mesurées
# - Prochaines étapes jour 2
```

#### **ÉTAPE 1.8 - Validation Performance (30 min)**
```bash
# Tests performance RTX 3090
python tests/test_prism_poc.py

# Validation standards GPU
python test_gpu_correct.py
python test_validation_rtx3090_detection.py

# Benchmark RTX 3090
python test_benchmark_performance_rtx3090.py
```

#### **ÉTAPE 1.7 - Documentation Jour 1 (1h)**
```bash
# Créer rapport journalier
echo "# JOUR 1 - PoC Prism RTX 3090" > docs/rapport_jour1.md
# Ajouter métriques, screenshots, résultats tests
```

**🎯 Livrable Jour 1 :** Backend PrismSTTBackend opérationnel RTX 3090 + Tests PoC validés

---

### **🚀 JOUR 2 - UNIFIEDSTTMANAGER RTX 3090**

#### **Matin (4h) - Architecture Manager**

#### **ÉTAPE 2.1 - UnifiedSTTManager (3h)**
**Créer :** `STT/unified_stt_manager.py`

```python
#!/usr/bin/env python3
"""
UnifiedSTTManager SuperWhisper V6 - RTX 3090 Unique
🚨 CONFIGURATION GPU: RTX 3090 (cuda:0) OBLIGATOIRE
"""

import os
# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

import asyncio
import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from STT.backends.prism_stt_backend import PrismSTTBackend
from STT.cache_manager import STTCache
from STT.circuit_breaker import CircuitBreaker
from STT.metrics import PrometheusSTTMetrics

class UnifiedSTTManager:
    """Manager STT unifié RTX 3090 - SuperWhisper V6"""
    
    def __init__(self):
        validate_rtx3090_mandatory()
        
        # Backends avec fallback intelligent  
        self.backends = {
            'prism_large': PrismSTTBackend(model_size='large-v2'),
            'prism_tiny': PrismSTTBackend(model_size='tiny'),
            'offline': OfflineSTTBackend()  # CPU fallback
        }
        
        # Fallback chain optimisé RTX 3090
        self.fallback_chain = ['prism_large', 'prism_tiny', 'offline']
        
        # Cache LRU (cohérent avec TTS Phase 3)
        self.cache = STTCache(max_size=200*1024*1024)  # 200MB
        
        # Circuit breakers par backend
        self.circuit_breakers = {
            name: CircuitBreaker(failure_threshold=3, recovery_timeout=30)
            for name in self.backends.keys()
        }
        
        # Métriques Prometheus
        self.metrics = PrometheusSTTMetrics()
        
        print("✅ UnifiedSTTManager RTX 3090 initialisé")
    
    def _generate_cache_key(self, audio: np.ndarray) -> str:
        """Génère clé cache pour audio"""
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        return f"stt_{audio_hash}_{len(audio)}"
    
    @asynccontextmanager
    async def _memory_management_context(self):
        """Context manager gestion mémoire RTX 3090"""
        try:
            # Cleanup avant traitement STT
            torch.cuda.empty_cache()
            yield
        finally:
            # Cleanup après traitement
            torch.cuda.empty_cache()
    
    async def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcription avec fallback automatique"""
        
        # Vérifier cache
        cache_key = self._generate_cache_key(audio)
        if cached_result := self.cache.get(cache_key):
            self.metrics.cache_hits.inc()
            return cached_result
        
        # Timeout dynamique (5s par minute d'audio)
        audio_duration = len(audio) / 16000
        timeout = max(5.0, audio_duration * 5)
        
        async with self._memory_management_context():
            # Tentative backends avec fallback
            for backend_name in self.fallback_chain:
                if self.circuit_breakers[backend_name].is_open():
                    continue
                
                try:
                    backend = self.backends[backend_name]
                    
                    result = await asyncio.wait_for(
                        backend.transcribe(audio),
                        timeout=timeout
                    )
                    
                    # Succès - mise en cache
                    self.cache.put(cache_key, result)
                    self.metrics.transcriptions_success.labels(backend=backend_name).inc()
                    
                    return result
                    
                except Exception as e:
                    self.circuit_breakers[backend_name].record_failure()
                    self.metrics.transcriptions_failed.labels(backend=backend_name).inc()
                    print(f"⚠️ Backend {backend_name} échec: {e}")
                    continue
        
        self.metrics.total_failures.inc()
        raise Exception("Tous les backends STT ont échoué")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne métriques système"""
        return {
            "cache_size": self.cache.current_size,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "backends_status": {
                name: "open" if cb.is_open() else "closed"
                for name, cb in self.circuit_breakers.items()
            }
        }
```

#### **ÉTAPE 2.2 - Cache STT (1h)**
**Créer :** `STT/cache_manager.py`

```python
class STTCache:
    """Cache LRU pour STT - Cohérent avec TTS Phase 3"""
    
    def __init__(self, max_size: int = 200*1024*1024):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            self.hits += 1
            self._update_access(key)
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Dict[str, Any]):
        # Estimation taille
        estimated_size = len(str(value).encode('utf-8'))
        
        # Éviction si nécessaire
        while (self.current_size + estimated_size > self.max_size 
               and self.access_order):
            self._evict_lru()
        
        self.cache[key] = value
        self.current_size += estimated_size
        self._update_access(key)
```

#### **Après-midi (4h) - Tests et Optimisation**

#### **ÉTAPE 2.3 - Tests UnifiedSTTManager (2h)**
**Créer :** `tests/test_unified_stt_manager.py`

```python
@pytest.mark.asyncio
async def test_stt_manager_fallback():
    """Test fallback chain STT"""
    manager = UnifiedSTTManager()
    
    # Audio test
    audio = generate_test_audio(duration=3.0)
    
    # Test transcription normale
    result = await manager.transcribe(audio)
    assert result is not None
    assert "text" in result
    assert result["latency_ms"] < 500  # RTX 3090
    
    # Test cache
    result2 = await manager.transcribe(audio)  # Cache hit
    assert manager.cache.hits > 0

@pytest.mark.asyncio  
async def test_stt_manager_stress():
    """Test stress 10 requêtes parallèles"""
    manager = UnifiedSTTManager()
    
    # 10 audios différents
    audios = [generate_test_audio(duration=2.0) for _ in range(10)]
    
    # Traitement parallèle
    tasks = [manager.transcribe(audio) for audio in audios]
    results = await asyncio.gather(*tasks)
    
    # Validation
    assert len(results) == 10
    for result in results:
        assert result["rtf"] < 1.5  # Performance RTX 3090
```

#### **ÉTAPE 2.4 - Circuit Breakers et Métriques (1h)**
**Créer :** `STT/circuit_breaker.py` et `STT/metrics.py`

#### **ÉTAPE 2.5 - Optimisation Mémoire RTX 3090 (1h)**
```python
class RTX3090MemoryManager:
    """Gestionnaire mémoire optimisé RTX 3090"""
    
    def __init__(self):
        self.total_vram = 24  # GB
        self.reserved_stt = 8   # GB pour modèles STT
        self.reserved_tts = 4   # GB pour TTS existant  
        self.available = 12     # GB disponible
        
    def monitor_memory_usage(self):
        """Monitoring utilisation VRAM"""
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": self.total_vram - reserved
        }
```

**🎯 Livrable Jour 2 :** UnifiedSTTManager complet + Tests validés + Cache LRU + Circuit breakers

---

### **🚀 JOUR 3 - PIPELINE COMPLET STT→LLM→TTS**

#### **Matin (4h) - Intégration Pipeline**

#### **ÉTAPE 3.1 - Pipeline Voice-to-Voice RTX 3090 (2h)**
**Modifier :** `run_assistant.py`

```python
#!/usr/bin/env python3
"""
Pipeline Voice-to-Voice SuperWhisper V6 - RTX 3090
🚨 CONFIGURATION GPU: RTX 3090 (cuda:0) OBLIGATOIRE
"""

import os
# Configuration GPU RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import asyncio
import time
import numpy as np
from typing import Dict, Any

from STT.unified_stt_manager import UnifiedSTTManager
from TTS.tts_manager import UnifiedTTSManager  # Existant Phase 3
from LLM.llm_manager import LLMManager  # À créer ou adapter

class VoiceToVoicePipeline:
    """Pipeline voix-à-voix complet RTX 3090"""
    
    def __init__(self):
        validate_rtx3090_mandatory()
        
        # Managers (STT nouveau + TTS existant)
        self.stt_manager = UnifiedSTTManager()
        self.tts_manager = UnifiedTTSManager()  # Phase 3
        self.llm_manager = LLMManager()
        
        print("✅ Pipeline Voice-to-Voice RTX 3090 initialisé")
    
    async def process_voice_to_voice(self, audio_buffer: np.ndarray) -> bytes:
        """Pipeline complet voix-à-voix < 730ms"""
        
        start_total = time.time()
        
        try:
            # ÉTAPE 1: STT sur RTX 3090 (< 400ms)
            start_stt = time.time()
            stt_result = await self.stt_manager.transcribe(audio_buffer)
            stt_latency = (time.time() - start_stt) * 1000
            
            # ÉTAPE 2: LLM (< 300ms)
            start_llm = time.time()
            llm_response = await self.llm_manager.generate_response(
                stt_result["text"]
            )
            llm_latency = (time.time() - start_llm) * 1000
            
            # ÉTAPE 3: TTS sur RTX 3090 (29.5ms existant)
            start_tts = time.time()
            audio_output = await self.tts_manager.synthesize(llm_response)
            tts_latency = (time.time() - start_tts) * 1000
            
            # Métriques pipeline
            total_latency = (time.time() - start_total) * 1000
            
            print(f"✅ Pipeline RTX 3090:")
            print(f"   STT: {stt_latency:.0f}ms")
            print(f"   LLM: {llm_latency:.0f}ms") 
            print(f"   TTS: {tts_latency:.0f}ms")
            print(f"   TOTAL: {total_latency:.0f}ms")
            
            # Validation performance < 730ms
            if total_latency > 730:
                print(f"⚠️ Pipeline lent: {total_latency:.0f}ms > 730ms")
            
            return audio_output
            
        except Exception as e:
            print(f"❌ Erreur pipeline: {e}")
            raise

# Point d'entrée principal
async def main():
    """Demo pipeline voice-to-voice"""
    
    pipeline = VoiceToVoicePipeline()
    
    # Audio test
    test_audio = generate_test_audio(duration=5.0)
    
    # Traitement
    result_audio = await pipeline.process_voice_to_voice(test_audio)
    
    print(f"✅ Pipeline terminé: {len(result_audio)} bytes audio")

if __name__ == "__main__":
    asyncio.run(main())
```

#### **ÉTAPE 3.2 - Configuration STT (1h)**
**Créer :** `config/stt.yaml`

```yaml
# Configuration STT SuperWhisper V6 - RTX 3090
stt:
  # Backends configuration
  backends:
    prism_large:
      model_size: "large-v2"
      device: "cuda:0"  # RTX 3090
      compute_type: "float16"
      enabled: true
      priority: 1
      
    prism_tiny:
      model_size: "tiny"
      device: "cuda:0"  # RTX 3090  
      compute_type: "float16"
      enabled: true
      priority: 2
      
    offline:
      enabled: true
      priority: 3
  
  # Cache configuration
  cache:
    max_size_mb: 200
    ttl_hours: 2
    
  # Performance targets RTX 3090
  performance:
    max_latency_ms: 400
    target_rtf: 1.0
    
  # Circuit breaker
  circuit_breaker:
    failure_threshold: 3
    recovery_timeout_s: 30
```

#### **ÉTAPE 3.3 - LLM Manager Adaptation (1h)**
**Créer/Adapter :** `LLM/llm_manager.py`

```python
class LLMManager:
    """Manager LLM pour pipeline RTX 3090"""
    
    def __init__(self):
        # LLM sur CPU ou RTX 3090 selon mémoire disponible
        self.device = "cpu"  # Ou cuda:0 si mémoire
        
    async def generate_response(self, text: str) -> str:
        """Génération réponse LLM"""
        # Simulation ou vraie implémentation
        await asyncio.sleep(0.1)  # 100ms simulation
        return f"Réponse à: {text}"
```

#### **Après-midi (4h) - Tests et Validation**

#### **ÉTAPE 3.4 - Tests Pipeline Complet (2h)**
**Créer :** `tests/test_pipeline_integration.py`

```python
@pytest.mark.asyncio
async def test_pipeline_voice_to_voice_rtx3090():
    """Test pipeline complet STT→LLM→TTS RTX 3090"""
    
    pipeline = VoiceToVoicePipeline()
    
    # Audio test 5 secondes
    audio_input = generate_test_audio(duration=5.0)
    
    # Pipeline complet
    start_time = time.time()
    audio_output = await pipeline.process_voice_to_voice(audio_input)
    total_time = (time.time() - start_time) * 1000
    
    # Validations
    assert audio_output is not None
    assert len(audio_output) > 0
    assert total_time < 730, f"Pipeline {total_time:.0f}ms > 730ms"
    
    print(f"✅ Pipeline RTX 3090: {total_time:.0f}ms")

@pytest.mark.asyncio
async def test_pipeline_stress_rtx3090():
    """Test stress pipeline 5 requêtes parallèles"""
    
    pipeline = VoiceToVoicePipeline()
    
    # 5 audios test
    audios = [generate_test_audio(duration=3.0) for _ in range(5)]
    
    # Traitement parallèle
    start_time = time.time()
    tasks = [pipeline.process_voice_to_voice(audio) for audio in audios]
    results = await asyncio.gather(*tasks)
    total_time = (time.time() - start_time) * 1000
    
    # Validations
    assert len(results) == 5
    assert all(result is not None for result in results)
    print(f"✅ Stress test RTX 3090: {total_time:.0f}ms pour 5 requêtes")
```

#### **ÉTAPE 3.5 - Tests Conditions Réelles (1h)**
**Créer :** `scripts/demo_pipeline_live.py`

```python
#!/usr/bin/env python3
"""Demo pipeline avec micro réel"""

import pyaudio
import numpy as np
from run_assistant import VoiceToVoicePipeline

def record_audio(duration: float = 5.0, sample_rate: int = 16000):
    """Enregistrement micro réel"""
    audio = pyaudio.PyAudio()
    
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024
    )
    
    print(f"🎤 Enregistrement {duration}s...")
    
    frames = []
    for _ in range(int(sample_rate * duration / 1024)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    stream.close()
    audio.terminate()
    
    return np.concatenate(frames)

async def demo_live_pipeline():
    """Demo live avec micro"""
    
    pipeline = VoiceToVoicePipeline()
    
    print("🎯 Demo Pipeline Live RTX 3090")
    print("Parlez dans le micro...")
    
    # Enregistrement
    audio = record_audio(duration=5.0)
    
    # Pipeline
    result = await pipeline.process_voice_to_voice(audio)
    
    # Lecture résultat
    # [Code lecture audio via haut-parleurs]
    
    print("✅ Demo terminé")

if __name__ == "__main__":
    asyncio.run(demo_live_pipeline())
```

#### **ÉTAPE 3.6 - Monitoring et Métriques (1h)**
**Créer :** `monitoring/stt_dashboard.py`

```python
#!/usr/bin/env python3
"""Dashboard monitoring STT temps réel"""

from prometheus_client import start_http_server, Histogram, Counter, Gauge
import time

# Métriques Prometheus STT
stt_latency = Histogram('stt_latency_seconds', 'Latence STT')
stt_requests = Counter('stt_requests_total', 'Requêtes STT')
stt_rtf = Gauge('stt_rtf', 'Real-Time Factor STT')
pipeline_latency = Histogram('pipeline_latency_seconds', 'Latence pipeline E2E')

def start_metrics_server():
    """Démarre serveur métriques Prometheus"""
    start_http_server(8000)
    print("✅ Métriques Prometheus: http://localhost:8000/metrics")

if __name__ == "__main__":
    start_metrics_server()
    
    # Garder serveur actif
    while True:
        time.sleep(1)
```

**🎯 Livrable Jour 3 :** Pipeline voice-to-voice complet < 730ms + Tests validés + Monitoring

---

## 📊 CRITÈRES DE SUCCÈS PHASE 4

### ✅ **Performance RTX 3090 (Obligatoire)**
- [x] **STT Latence** : < 400ms pour 5s audio (Prism large-v2)
- [x] **STT RTF** : < 1.0 (temps réel)
- [x] **Pipeline Total** : < 730ms (STT + LLM + TTS)
- [x] **Cache Hit Rate** : > 30% (cohérent TTS 93.1%)
- [x] **Stabilité** : > 99% (comme TTS 100%)

### ✅ **Tests Validation (Obligatoire)**
- [x] **test_prism_poc.py** : PoC Prism RTX 3090 validé
- [x] **test_unified_stt_manager.py** : Manager coverage ≥ 90%
- [x] **test_pipeline_integration.py** : Pipeline E2E validé
- [x] **Tests stress** : 5 requêtes parallèles réussies
- [x] **Demo live** : Micro réel + haut-parleurs fonctionnels

### ✅ **Standards GPU (Obligatoire)**
- [x] **Configuration RTX 3090** : Tous fichiers conformes
- [x] **Validation `validate_rtx3090_mandatory()`** : Systématique
- [x] **Tests GPU** : `test_gpu_correct.py` validé
- [x] **Pas d'utilisation RTX 5060 Ti** : 0% risque

### ✅ **Architecture (Obligatoire)**
- [x] **Cohérence avec TTS** : Pattern similaire Phase 3
- [x] **Fallback intelligent** : prism_large → prism_tiny → offline
- [x] **Cache LRU** : 200MB cohérent avec TTS
- [x] **Circuit breakers** : Protection robuste
- [x] **Métriques Prometheus** : Monitoring professionnel

---

## 🚀 LIVRABLES FINAUX

### **Code Principal**
```
STT/
├── backends/
│   ├── prism_stt_backend.py      # Backend Prism RTX 3090
│   └── offline_stt_backend.py    # Fallback CPU
├── unified_stt_manager.py        # Manager unifié
├── cache_manager.py              # Cache LRU STT
├── circuit_breaker.py            # Protection robustesse
└── metrics.py                    # Métriques Prometheus
```

### **Tests**
```
tests/
├── test_prism_poc.py             # PoC validation RTX 3090
├── test_unified_stt_manager.py   # Tests manager complet
├── test_pipeline_integration.py  # Pipeline STT→LLM→TTS
└── test_stt_stress.py           # Tests charge
```

### **Configuration et Scripts**
```
config/
└── stt.yaml                      # Configuration STT

scripts/
├── demo_pipeline_live.py         # Demo micro réel
└── validate_dual_gpu_rtx3090.py  # Validation GPU

monitoring/
└── stt_dashboard.py              # Dashboard métriques
```

### **Documentation**
```
docs/
├── rapport_jour1.md              # PoC validation
├── rapport_jour2.md              # Manager et tests
├── rapport_jour3.md              # Pipeline et démo
└── phase4_stt_complete.md        # Documentation finale
```

---

## 🎯 MÉTRIQUES SUCCÈS FINALES

### **Performance Atteinte (Cible)**
- **STT Prism Large** : < 400ms RTX 3090 ✅
- **Pipeline E2E** : < 730ms total ✅
- **Cache Hit Rate** : > 30% ✅
- **Tests Coverage** : > 90% ✅
- **Stabilité** : > 99% ✅

### **Validation Standards**
- **Configuration GPU** : 100% conforme ✅
- **Tests GPU** : Validés ✅
- **Architecture** : Cohérente Phase 3 ✅
- **Documentation** : Complète ✅

---

**🎯 AVEC CE PLAN, LIVREZ UN PIPELINE VOICE-TO-VOICE PROFESSIONNEL !**  
**🚀 RTX 3090 OPTIMISÉ + PERFORMANCE RECORD + STANDARDS SUPERWHISPER V6**

---

*Plan créé le 12/06/2025 - Phase 4 STT SuperWhisper V6*  
*Configuration : RTX 3090 Unique (24GB VRAM) - 3 jours* 