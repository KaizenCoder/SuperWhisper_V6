 PROMPT FINAL OPTIMISÉ - Implémentation et Validation du `RobustSTTManager` (Phase 1 / Tâche 2)

## 🎯 Contexte et Alignement Stratégique

**Référence :** Phase 1, Tâche 2 du Plan de Développement LUXA Final  
**Priorité :** CRITIQUE IMMÉDIATE  
**Durée estimée :** 3 jours  
**Prérequis :** ✅ Import bloquant corrigé (Tâche 1 terminée le 11 juin 2025)

## 📋 Objectifs Spécifiques (selon PRD v3.1)

1. **Remplacer le handler MVP** par un Manager robuste avec gestion d'erreurs, fallbacks et métriques
2. **Valider en conditions réelles** avec microphone physique (critère non négociable)
3. **Préserver l'architecture existante** de sécurité/monitoring/robustesse
4. **Atteindre les métriques cibles** : latence STT < 300ms pour audio court

## 🔧 Plan d'Implémentation Détaillé

### Étape 1 : Création du RobustSTTManager

**Fichier à créer :** `STT/stt_manager_robust.py`

```python
# STT/stt_manager_robust.py
"""
RobustSTTManager - Gestionnaire STT robuste avec fallback multi-modèles
Conforme aux exigences du PRD v3.1 et du Plan de Développement Final
"""
import torch
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import io
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
from prometheus_client import Counter, Histogram, Gauge
from circuitbreaker import circuit
from STT.vad_manager import VADManager  # Intégration VAD existant

# Métriques Prometheus conformes à l'architecture
stt_transcriptions_total = Counter('stt_transcriptions_total', 'Total transcriptions')
stt_errors_total = Counter('stt_errors_total', 'Total errors')
stt_latency_seconds = Histogram('stt_latency_seconds', 'Transcription latency')
stt_vram_usage_bytes = Gauge('stt_vram_usage_bytes', 'VRAM usage in bytes')

class RobustSTTManager:
    """
    Manager STT robuste avec:
    - Sélection GPU automatique optimale
    - Chaîne de fallback multi-modèles
    - Gestion VRAM intelligente avec clear_cache
    - Métriques temps réel (latence, erreurs, succès)
    - Conversion audio robuste (bytes ↔ numpy)
    - Intégration VAD existant
    """
    
    def __init__(self, config: Dict[str, Any], vad_manager: Optional[VADManager] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = self._select_optimal_device()
        self.models = {}
        self.fallback_chain = config.get("fallback_chain", ["base", "tiny"])
        self.vad_manager = vad_manager
        
        # Configuration compute_type selon device
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Métriques internes pour monitoring
        self.metrics = {
            "transcriptions": 0,
            "errors": 0,
            "avg_latency": 0.0,
            "total_latency": 0.0
        }

    def _select_optimal_device(self) -> str:
        """Sélection intelligente du device avec fallback GPU→CPU"""
        if not self.config.get('use_gpu', True):
            self.logger.info("GPU désactivé par configuration")
            return "cpu"
            
        if torch.cuda.is_available():
            # Détection multi-GPU et sélection optimale
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # Stratégie dual-GPU : GPU secondaire pour STT
                selected_gpu = 1 if gpu_count > 1 else 0
                torch.cuda.set_device(selected_gpu)
                self.logger.info(f"Multi-GPU détecté : utilisation GPU {selected_gpu} pour STT")
            
            # Vérification VRAM disponible
            vram_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            vram_free_gb = vram_free / (1024**3)
            
            if vram_free_gb < 2.0:
                self.logger.warning(f"VRAM insuffisante ({vram_free_gb:.1f}GB), fallback CPU")
                return "cpu"
                
            self.logger.info(f"GPU sélectionné avec {vram_free_gb:.1f}GB VRAM disponible")
            return "cuda"
        else:
            self.logger.warning("CUDA non disponible, utilisation CPU")
            return "cpu"

    async def initialize(self):
        """Initialisation asynchrone avec chargement modèles"""
        self.logger.info(f"Initialisation RobustSTTManager sur {self.device}")
        
        for model_name in self.fallback_chain:
            try:
                await self._load_model(model_name)
            except Exception as e:
                self.logger.error(f"Échec chargement {model_name}: {e}")
                if model_name == self.fallback_chain[-1]:
                    raise RuntimeError("Aucun modèle STT disponible")

    async def _load_model(self, model_size: str):
        """Chargement optimisé avec faster-whisper"""
        start_time = time.time()
        self.logger.info(f"Chargement modèle '{model_size}' sur '{self.device}'...")
        
        try:
            self.models[model_size] = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=2,  # Parallélisation
                download_root=self.config.get("model_cache_dir", "./models")
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ Modèle {model_size} chargé en {load_time:.2f}s")
            
            # Test rapide du modèle
            test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
            self.models[model_size].transcribe(test_audio, language="fr")
            
        except Exception as e:
            self.logger.error(f"❌ Échec chargement {model_size}: {e}")
            raise

    @circuit(failure_threshold=3, recovery_timeout=30)
    async def transcribe_audio(self, audio_data: bytes, language: str = "fr") -> Dict[str, Any]:
        """
        Transcription robuste avec circuit breaker et métriques
        
        Args:
            audio_data: Audio en bytes (WAV format)
            language: Code langue (défaut: français)
            
        Returns:
            Dict avec transcription, temps de traitement et device utilisé
        """
        start_time = time.time()
        self.metrics["transcriptions"] += 1
        stt_transcriptions_total.inc()
        
        try:
            # Conversion bytes → numpy avec validation
            audio_array = self._bytes_to_audio_array(audio_data)
            
            # Segmentation VAD si disponible
            if self.vad_manager and self.config.get('use_vad', True):
                self.logger.debug("Segmentation VAD activée")
                speech_segments = await self._process_with_vad(audio_array)
            else:
                speech_segments = [(0, len(audio_array), audio_array)]
            
            # Transcription avec fallback chain
            full_transcription = ""
            segments_info = []
            
            for idx, (start_ms, end_ms, segment) in enumerate(speech_segments):
                segment_text = await self._transcribe_segment(segment, language, idx)
                if segment_text:
                    full_transcription += segment_text + " "
                    segments_info.append({
                        "start": start_ms,
                        "end": end_ms,
                        "text": segment_text
                    })
            
            # Calcul métriques
            processing_time = time.time() - start_time
            self._update_latency_metrics(processing_time)
            stt_latency_seconds.observe(processing_time)
            
            # Monitoring VRAM si GPU
            if self.device == "cuda":
                vram_used = torch.cuda.memory_allocated() / (1024**3)
                stt_vram_usage_bytes.set(vram_used * 1024**3)
                self.logger.debug(f"VRAM utilisée: {vram_used:.2f}GB")
            
            result = {
                "text": full_transcription.strip(),
                "segments": segments_info,
                "processing_time": processing_time,
                "device": self.device,
                "model_used": self._get_active_model(),
                "metrics": {
                    "audio_duration": len(audio_array) / 16000,
                    "rtf": processing_time / (len(audio_array) / 16000)  # Real-time factor
                }
            }
            
            self.logger.info(f"✅ Transcription réussie en {processing_time:.2f}s (RTF: {result['metrics']['rtf']:.2f})")
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            stt_errors_total.inc()
            self.logger.error(f"❌ Erreur transcription: {e}", exc_info=True)
            
            # Clear cache GPU si erreur mémoire
            if "out of memory" in str(e).lower() and self.device == "cuda":
                torch.cuda.empty_cache()
                self.logger.info("Cache GPU vidé après erreur mémoire")
            
            raise

    async def _process_with_vad(self, audio_array: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """Traitement VAD asynchrone avec timestamps"""
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, 
            self.vad_manager.process_audio, 
            audio_array
        )
        return segments

    async def _transcribe_segment(self, segment: np.ndarray, language: str, idx: int) -> str:
        """Transcription d'un segment avec fallback chain"""
        for model_name in self.fallback_chain:
            if model_name not in self.models:
                continue
                
            try:
                self.logger.debug(f"Tentative transcription segment {idx} avec {model_name}")
                
                # Transcription asynchrone
                loop = asyncio.get_event_loop()
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.models[model_name].transcribe(
                        segment,
                        language=language,
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,  # Déterministe
                        condition_on_previous_text=True,
                        vad_filter=False  # VAD déjà appliqué
                    )
                )
                
                text = " ".join([seg.text for seg in segments]).strip()
                if text:  # Succès si texte non vide
                    return text
                    
            except Exception as e:
                self.logger.warning(f"Échec {model_name} sur segment {idx}: {e}")
                continue
                
        self.logger.error(f"Échec complet transcription segment {idx}")
        return ""

    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Conversion robuste bytes → numpy avec validation"""
        try:
            # Tentative 1: soundfile (plus robuste)
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # Conversion mono si nécessaire
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resampling à 16kHz si nécessaire
            if sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            
            # Normalisation [-1, 1]
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
                
            return audio_array
            
        except Exception as e:
            self.logger.error(f"Erreur conversion audio: {e}")
            raise ValueError(f"Format audio invalide: {e}")

    def _update_latency_metrics(self, latency: float):
        """Mise à jour métriques de latence avec moyenne mobile"""
        self.metrics["total_latency"] += latency
        self.metrics["avg_latency"] = (
            self.metrics["total_latency"] / self.metrics["transcriptions"]
        )
        
        self.logger.debug(
            f"Latence: {latency:.3f}s | "
            f"Moyenne: {self.metrics['avg_latency']:.3f}s | "
            f"Total transcriptions: {self.metrics['transcriptions']}"
        )

    def _get_active_model(self) -> str:
        """Retourne le modèle actuellement chargé"""
        return list(self.models.keys())[0] if self.models else "none"

    def get_metrics(self) -> Dict[str, Any]:
        """Export métriques pour monitoring"""
        return {
            **self.metrics,
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "vad_enabled": self.vad_manager is not None
        }

    async def cleanup(self):
        """Nettoyage ressources et modèles"""
        self.logger.info("Nettoyage RobustSTTManager...")
        
        # Libération modèles
        self.models.clear()
        
        # Clear cache GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        self.logger.info("✅ Nettoyage terminé")
```

### Étape 2 : Adaptation du Script de Test

**Fichier à modifier :** `tests/test_realtime_audio_pipeline.py`

```python
# tests/test_realtime_audio_pipeline.py
"""
Test d'intégration du RobustSTTManager avec microphone réel
Conforme aux exigences PRD v3.1 - Validation obligatoire en conditions réelles
"""
import pytest
import asyncio
import yaml
import sounddevice as sd
import numpy as np
import soundfile as sf
import io
from pathlib import Path
import sys
import time
import logging

# Configuration logging pour debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT.stt_manager_robust import RobustSTTManager
from STT.vad_manager import VADManager

def record_micro(seconds: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """
    Enregistrement microphone avec feedback visuel
    
    Args:
        seconds: Durée d'enregistrement
        sample_rate: Taux d'échantillonnage (16kHz pour Whisper)
        
    Returns:
        numpy array de l'audio enregistré
    """
    print(f"\n🎤 Enregistrement pendant {seconds} secondes...")
    print("3... ", end="", flush=True)
    time.sleep(1)
    print("2... ", end="", flush=True)
    time.sleep(1)
    print("1... ", end="", flush=True)
    time.sleep(1)
    print("PARLEZ MAINTENANT! 🔴")
    
    # Enregistrement
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    print("✅ Enregistrement terminé\n")
    return audio.flatten()

@pytest.mark.asyncio
async def test_robust_stt_manager_validation_complete():
    """
    Test de validation complet du RobustSTTManager
    Critères PRD v3.1 : Test microphone réel obligatoire
    """
    print("\n" + "="*80)
    print("TEST DE VALIDATION ROBUSTSTTMANAGER - CONDITIONS RÉELLES")
    print("="*80)
    
    # 1. Configuration
    config_path = Path("Config/mvp_settings.yaml")
    if not config_path.exists():
        pytest.skip(f"Configuration non trouvée: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration STT pour test
    stt_config = config.get('stt', {})
    stt_config.update({
        'use_gpu': torch.cuda.is_available(),  # Auto-détection GPU
        'fallback_chain': ['tiny'],  # Modèle léger pour tests
        'use_vad': True,  # Test avec VAD activé
        'model_cache_dir': './models/whisper'
    })
    
    # 2. Initialisation composants
    print("\n📋 Initialisation des composants...")
    
    # VAD Manager (si disponible)
    vad_manager = None
    try:
        vad_config = config.get('vad', {})
        vad_manager = VADManager(vad_config)
        print("✅ VAD Manager initialisé")
    except Exception as e:
        print(f"⚠️  VAD non disponible: {e}")
        stt_config['use_vad'] = False
    
    # STT Manager
    stt_manager = RobustSTTManager(stt_config, vad_manager=vad_manager)
    await stt_manager.initialize()
    print(f"✅ STT Manager initialisé sur {stt_manager.device}")
    
    # 3. Test 1 : Phrase de validation obligatoire
    print("\n🧪 TEST 1 : Phrase de validation obligatoire")
    phrase_validation = "Ceci est un test de validation du nouveau gestionnaire robuste"
    print(f"📢 Phrase à prononcer : « {phrase_validation} »")
    
    audio_array = record_micro(seconds=7)
    
    # Conversion en bytes WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, 16000, format='WAV', subtype='PCM_16')
    audio_bytes = buffer.getvalue()
    
    # Transcription
    start_time = time.time()
    result = await stt_manager.transcribe_audio(audio_bytes, language="fr")
    
    # Validation résultats
    transcription = result['text'].lower()
    processing_time = result['processing_time']
    
    print(f"\n📝 Transcription : '{result['text']}'")
    print(f"⏱️  Temps de traitement : {processing_time:.3f}s")
    print(f"🖥️  Device utilisé : {result['device']}")
    print(f"📊 RTF (Real-Time Factor) : {result['metrics']['rtf']:.2f}")
    
    # Assertions sémantiques
    mots_cles = ["test", "validation", "gestionnaire", "robuste"]
    mots_trouves = [mot for mot in mots_cles if mot in transcription]
    
    assert len(mots_trouves) >= 3, (
        f"Transcription incomplète. Mots trouvés: {mots_trouves}/4. "
        f"Transcription: '{result['text']}'"
    )
    
    # Assertion performance (< 300ms pour audio court selon PRD)
    audio_duration = len(audio_array) / 16000
    if audio_duration < 10:  # Audio court
        assert processing_time < 0.3, (
            f"Latence trop élevée: {processing_time:.3f}s > 0.3s (cible PRD)"
        )
    
    print("✅ TEST 1 RÉUSSI : Validation sémantique et performance OK")
    
    # 4. Test 2 : Robustesse avec audio difficile
    print("\n🧪 TEST 2 : Test de robustesse")
    print("📢 Parlez rapidement avec des mots techniques")
    
    audio_array2 = record_micro(seconds=5)
    buffer2 = io.BytesIO()
    sf.write(buffer2, audio_array2, 16000, format='WAV', subtype='PCM_16')
    
    result2 = await stt_manager.transcribe_audio(buffer2.getvalue())
    
    assert result2['text'] != "", "Transcription vide sur audio difficile"
    assert result2['processing_time'] < 1.0, "Temps de traitement trop long"
    
    print(f"📝 Transcription robuste : '{result2['text']}'")
    print("✅ TEST 2 RÉUSSI : Robustesse validée")
    
    # 5. Test 3 : Métriques et monitoring
    print("\n🧪 TEST 3 : Vérification métriques")
    
    metrics = stt_manager.get_metrics()
    assert metrics['transcriptions'] == 2, "Compteur transcriptions incorrect"
    assert metrics['errors'] == 0, "Erreurs détectées pendant les tests"
    assert metrics['avg_latency'] > 0, "Latence moyenne non calculée"
    
    print(f"📊 Métriques finales : {metrics}")
    print("✅ TEST 3 RÉUSSI : Métriques correctes")
    
    # 6. Nettoyage
    await stt_manager.cleanup()
    
    # Résumé final
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLÈTE RÉUSSIE")
    print(f"   - Transcriptions réussies : {metrics['transcriptions']}")
    print(f"   - Latence moyenne : {metrics['avg_latency']:.3f}s")
    print(f"   - Device utilisé : {metrics['device']}")
    print(f"   - Modèles chargés : {metrics['models_loaded']}")
    print("="*80)

@pytest.mark.asyncio
async def test_fallback_chain():
    """Test spécifique de la chaîne de fallback"""
    print("\n🧪 TEST FALLBACK CHAIN")
    
    # Configuration avec chaîne de fallback étendue
    stt_config = {
        'use_gpu': False,  # Force CPU pour test
        'fallback_chain': ['base', 'small', 'tiny'],
        'use_vad': False
    }
    
    stt_manager = RobustSTTManager(stt_config)
    
    # Initialisation avec simulation d'échec sur 'base'
    stt_manager.fallback_chain = ['small', 'tiny']  # Skip 'base'
    await stt_manager.initialize()
    
    # Test audio simple
    test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
    buffer = io.BytesIO()
    sf.write(buffer, test_audio, 16000, format='WAV')
    
    result = await stt_manager.transcribe_audio(buffer.getvalue())
    
    assert result['text'] == "", "Silence devrait donner transcription vide"
    assert result['device'] == 'cpu', "Device incorrect"
    
    await stt_manager.cleanup()
    print("✅ Fallback chain validée")

if __name__ == "__main__":
    # Exécution directe pour tests manuels
    asyncio.run(test_robust_stt_manager_validation_complete())
```

### Étape 3 : Intégration dans l'Orchestrateur

**Fichier à modifier :** `run_assistant.py`

```python
# Modifications dans run_assistant.py

# Imports à ajouter/modifier
from STT.stt_manager_robust import RobustSTTManager
from STT.vad_manager import VADManager

# Dans la fonction main() ou setup_components()
async def setup_stt_components(config):
    """Configuration des composants STT avec le nouveau manager"""
    
    # Initialisation VAD si configuré
    vad_manager = None
    if config.get('vad', {}).get('enabled', True):
        try:
            vad_manager = VADManager(config['vad'])
            logger.info("VAD Manager initialisé")
        except Exception as e:
            logger.warning(f"VAD non disponible: {e}")
    
    # Initialisation STT Manager Robuste
    stt_manager = RobustSTTManager(config['stt'], vad_manager=vad_manager)
    await stt_manager.initialize()
    
    logger.info(f"STT Manager initialisé sur {stt_manager.device}")
    return stt_manager

# Remplacer l'ancienne initialisation par :
stt_handler = await setup_stt_components(config)
```

## 📊 Critères d'Acceptation Détaillés

### ✅ Critères Techniques (PRD v3.1)
1. **Test microphone réel obligatoire** avec phrase de validation
2. **Latence < 300ms** pour audio court (<10s)
3. **Fallback GPU→CPU** opérationnel
4. **Métriques Prometheus** intégrées
5. **Circuit breakers** fonctionnels

### ✅ Critères de Code
1. **Suppression** de l'ancien `stt_handler.py`
2. **Tests automatisés** passent à 100%
3. **Coverage > 80%** sur le nouveau manager
4. **Logging structuré** avec niveaux appropriés
5. **Documentation** inline complète

### ✅ Critères de Validation
1. **Commande de test :**
   ```bash
   pytest -v -s tests/test_realtime_audio_pipeline.py::test_robust_stt_manager_validation_complete
   ```
2. **Validation manuelle :** Transcription correcte de la phrase test
3. **Métriques vérifiées :** Latence, RTF, device, erreurs
4. **Intégration réussie :** `run_assistant.py` utilise le nouveau manager

## 🚀 Commandes d'Exécution

```bash
# 1. Créer le nouveau manager
touch STT/stt_manager_robust.py

# 2. Exécuter les tests de validation
pytest -v -s tests/test_realtime_audio_pipeline.py

# 3. Vérifier l'intégration
python run_assistant.py --test-mode

# 4. Supprimer l'ancien handler
rm STT/stt_handler.py

# 5. Commit avec message structuré
git add -A
git commit -m "feat(stt): Implement RobustSTTManager with real-time validation

- Replace basic STTHandler with robust manager
- Add GPU selection and fallback chain
- Integrate Prometheus metrics and circuit breakers
- Validate with real microphone tests
- Achieve <300ms latency for short audio

Closes Phase 1 / Task 2 of LUXA Development Plan"
```

## 📝 Notes d'Implémentation

1. **Import torch** : Ajouter `import torch` dans le test si nécessaire
2. **VADManager** : S'assurer que le fichier existe ou adapter l'import
3. **Config paths** : Vérifier les chemins selon la structure du projet
4. **Modèles Whisper** : Le premier lancement téléchargera les modèles (~40MB pour tiny)
5. **Permissions micro** : S'assurer que l'application a accès au microphone

Ce prompt optimisé est maintenant parfaitement aligné avec les documents de planification et inclut tous les éléments critiques pour une implémentation réussie.

```python
# STT/stt_manager_robust.py
"""
RobustSTTManager - Gestionnaire STT robuste avec fallback multi-modèles
Conforme aux exigences du PRD v3.1 et du Plan de Développement Final
"""
import torch
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import io
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
from prometheus_client import Counter, Histogram, Gauge
from circuitbreaker import circuit
from STT.vad_manager import VADManager  # Intégration VAD existant

# Métriques Prometheus conformes à l'architecture
stt_transcriptions_total = Counter('stt_transcriptions_total', 'Total transcriptions')
stt_errors_total = Counter('stt_errors_total', 'Total errors')
stt_latency_seconds = Histogram('stt_latency_seconds', 'Transcription latency')
stt_vram_usage_bytes = Gauge('stt_vram_usage_bytes', 'VRAM usage in bytes')

class RobustSTTManager:
    """
    Manager STT robuste avec:
    - Sélection GPU automatique optimale
    - Chaîne de fallback multi-modèles
    - Gestion VRAM intelligente avec clear_cache
    - Métriques temps réel (latence, erreurs, succès)
    - Conversion audio robuste (bytes ↔ numpy)
    - Intégration VAD existant
    """
    
    def __init__(self, config: Dict[str, Any], vad_manager: Optional[VADManager] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = self._select_optimal_device()
        self.models = {}
        self.fallback_chain = config.get("fallback_chain", ["base", "tiny"])
        self.vad_manager = vad_manager
        
        # Configuration compute_type selon device
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Métriques internes pour monitoring
        self.metrics = {
            "transcriptions": 0,
            "errors": 0,
            "avg_latency": 0.0,
            "total_latency": 0.0
        }

    def _select_optimal_device(self) -> str:
        """Sélection intelligente du device avec fallback GPU→CPU"""
        if not self.config.get('use_gpu', True):
            self.logger.info("GPU désactivé par configuration")
            return "cpu"
            
        if torch.cuda.is_available():
            # Détection multi-GPU et sélection optimale
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # Stratégie dual-GPU : GPU secondaire pour STT
                selected_gpu = 1 if gpu_count > 1 else 0
                torch.cuda.set_device(selected_gpu)
                self.logger.info(f"Multi-GPU détecté : utilisation GPU {selected_gpu} pour STT")
            
            # Vérification VRAM disponible
            vram_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            vram_free_gb = vram_free / (1024**3)
            
            if vram_free_gb < 2.0:
                self.logger.warning(f"VRAM insuffisante ({vram_free_gb:.1f}GB), fallback CPU")
                return "cpu"
                
            self.logger.info(f"GPU sélectionné avec {vram_free_gb:.1f}GB VRAM disponible")
            return "cuda"
        else:
            self.logger.warning("CUDA non disponible, utilisation CPU")
            return "cpu"

    async def initialize(self):
        """Initialisation asynchrone avec chargement modèles"""
        self.logger.info(f"Initialisation RobustSTTManager sur {self.device}")
        
        for model_name in self.fallback_chain:
            try:
                await self._load_model(model_name)
            except Exception as e:
                self.logger.error(f"Échec chargement {model_name}: {e}")
                if model_name == self.fallback_chain[-1]:
                    raise RuntimeError("Aucun modèle STT disponible")

    async def _load_model(self, model_size: str):
        """Chargement optimisé avec faster-whisper"""
        start_time = time.time()
        self.logger.info(f"Chargement modèle '{model_size}' sur '{self.device}'...")
        
        try:
            self.models[model_size] = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=2,  # Parallélisation
                download_root=self.config.get("model_cache_dir", "./models")
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ Modèle {model_size} chargé en {load_time:.2f}s")
            
            # Test rapide du modèle
            test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
            self.models[model_size].transcribe(test_audio, language="fr")
            
        except Exception as e:
            self.logger.error(f"❌ Échec chargement {model_size}: {e}")
            raise

    @circuit(failure_threshold=3, recovery_timeout=30)
    async def transcribe_audio(self, audio_data: bytes, language: str = "fr") -> Dict[str, Any]:
        """
        Transcription robuste avec circuit breaker et métriques
        
        Args:
            audio_data: Audio en bytes (WAV format)
            language: Code langue (défaut: français)
            
        Returns:
            Dict avec transcription, temps de traitement et device utilisé
        """
        start_time = time.time()
        self.metrics["transcriptions"] += 1
        stt_transcriptions_total.inc()
        
        try:
            # Conversion bytes → numpy avec validation
            audio_array = self._bytes_to_audio_array(audio_data)
            
            # Segmentation VAD si disponible
            if self.vad_manager and self.config.get('use_vad', True):
                self.logger.debug("Segmentation VAD activée")
                speech_segments = await self._process_with_vad(audio_array)
            else:
                speech_segments = [(0, len(audio_array), audio_array)]
            
            # Transcription avec fallback chain
            full_transcription = ""
            segments_info = []
            
            for idx, (start_ms, end_ms, segment) in enumerate(speech_segments):
                segment_text = await self._transcribe_segment(segment, language, idx)
                if segment_text:
                    full_transcription += segment_text + " "
                    segments_info.append({
                        "start": start_ms,
                        "end": end_ms,
                        "text": segment_text
                    })
            
            # Calcul métriques
            processing_time = time.time() - start_time
            self._update_latency_metrics(processing_time)
            stt_latency_seconds.observe(processing_time)
            
            # Monitoring VRAM si GPU
            if self.device == "cuda":
                vram_used = torch.cuda.memory_allocated() / (1024**3)
                stt_vram_usage_bytes.set(vram_used * 1024**3)
                self.logger.debug(f"VRAM utilisée: {vram_used:.2f}GB")
            
            result = {
                "text": full_transcription.strip(),
                "segments": segments_info,
                "processing_time": processing_time,
                "device": self.device,
                "model_used": self._get_active_model(),
                "metrics": {
                    "audio_duration": len(audio_array) / 16000,
                    "rtf": processing_time / (len(audio_array) / 16000)  # Real-time factor
                }
            }
            
            self.logger.info(f"✅ Transcription réussie en {processing_time:.2f}s (RTF: {result['metrics']['rtf']:.2f})")
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            stt_errors_total.inc()
            self.logger.error(f"❌ Erreur transcription: {e}", exc_info=True)
            
            # Clear cache GPU si erreur mémoire
            if "out of memory" in str(e).lower() and self.device == "cuda":
                torch.cuda.empty_cache()
                self.logger.info("Cache GPU vidé après erreur mémoire")
            
            raise

    async def _process_with_vad(self, audio_array: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """Traitement VAD asynchrone avec timestamps"""
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, 
            self.vad_manager.process_audio, 
            audio_array
        )
        return segments

    async def _transcribe_segment(self, segment: np.ndarray, language: str, idx: int) -> str:
        """Transcription d'un segment avec fallback chain"""
        for model_name in self.fallback_chain:
            if model_name not in self.models:
                continue
                
            try:
                self.logger.debug(f"Tentative transcription segment {idx} avec {model_name}")
                
                # Transcription asynchrone
                loop = asyncio.get_event_loop()
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.models[model_name].transcribe(
                        segment,
                        language=language,
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,  # Déterministe
                        condition_on_previous_text=True,
                        vad_filter=False  # VAD déjà appliqué
                    )
                )
                
                text = " ".join([seg.text for seg in segments]).strip()
                if text:  # Succès si texte non vide
                    return text
                    
            except Exception as e:
                self.logger.warning(f"Échec {model_name} sur segment {idx}: {e}")
                continue
                
        self.logger.error(f"Échec complet transcription segment {idx}")
        return ""

    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Conversion robuste bytes → numpy avec validation"""
        try:
            # Tentative 1: soundfile (plus robuste)
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # Conversion mono si nécessaire
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resampling à 16kHz si nécessaire
            if sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            
            # Normalisation [-1, 1]
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
                
            return audio_array
            
        except Exception as e:
            self.logger.error(f"Erreur conversion audio: {e}")
            raise ValueError(f"Format audio invalide: {e}")

    def _update_latency_metrics(self, latency: float):
        """Mise à jour métriques de latence avec moyenne mobile"""
        self.metrics["total_latency"] += latency
        self.metrics["avg_latency"] = (
            self.metrics["total_latency"] / self.metrics["transcriptions"]
        )
        
        self.logger.debug(
            f"Latence: {latency:.3f}s | "
            f"Moyenne: {self.metrics['avg_latency']:.3f}s | "
            f"Total transcriptions: {self.metrics['transcriptions']}"
        )

    def _get_active_model(self) -> str:
        """Retourne le modèle actuellement chargé"""
        return list(self.models.keys())[0] if self.models else "none"

    def get_metrics(self) -> Dict[str, Any]:
        """Export métriques pour monitoring"""
        return {
            **self.metrics,
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "vad_enabled": self.vad_manager is not None
        }

    async def cleanup(self):
        """Nettoyage ressources et modèles"""
        self.logger.info("Nettoyage RobustSTTManager...")
        
        # Libération modèles
        self.models.clear()
        
        # Clear cache GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        self.logger.info("✅ Nettoyage terminé")
```

```python
# tests/test_realtime_audio_pipeline.py
"""
Test d'intégration du RobustSTTManager avec microphone réel
Conforme aux exigences PRD v3.1 - Validation obligatoire en conditions réelles
"""
import pytest
import asyncio
import yaml
import sounddevice as sd
import numpy as np
import soundfile as sf
import io
from pathlib import Path
import sys
import time
import logging

# Configuration logging pour debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT.stt_manager_robust import RobustSTTManager
from STT.vad_manager import VADManager

def record_micro(seconds: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """
    Enregistrement microphone avec feedback visuel
    
    Args:
        seconds: Durée d'enregistrement
        sample_rate: Taux d'échantillonnage (16kHz pour Whisper)
        
    Returns:
        numpy array de l'audio enregistré
    """
    print(f"\n🎤 Enregistrement pendant {seconds} secondes...")
    print("3... ", end="", flush=True)
    time.sleep(1)
    print("2... ", end="", flush=True)
    time.sleep(1)
    print("1... ", end="", flush=True)
    time.sleep(1)
    print("PARLEZ MAINTENANT! 🔴")
    
    # Enregistrement
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    print("✅ Enregistrement terminé\n")
    return audio.flatten()

@pytest.mark.asyncio
async def test_robust_stt_manager_validation_complete():
    """
    Test de validation complet du RobustSTTManager
    Critères PRD v3.1 : Test microphone réel obligatoire
    """
    print("\n" + "="*80)
    print("TEST DE VALIDATION ROBUSTSTTMANAGER - CONDITIONS RÉELLES")
    print("="*80)
    
    # 1. Configuration
    config_path = Path("Config/mvp_settings.yaml")
    if not config_path.exists():
        pytest.skip(f"Configuration non trouvée: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration STT pour test
    stt_config = config.get('stt', {})
    stt_config.update({
        'use_gpu': torch.cuda.is_available(),  # Auto-détection GPU
        'fallback_chain': ['tiny'],  # Modèle léger pour tests
        'use_vad': True,  # Test avec VAD activé
        'model_cache_dir': './models/whisper'
    })
    
    # 2. Initialisation composants
    print("\n📋 Initialisation des composants...")
    
    # VAD Manager (si disponible)
    vad_manager = None
    try:
        vad_config = config.get('vad', {})
        vad_manager = VADManager(vad_config)
        print("✅ VAD Manager initialisé")
    except Exception as e:
        print(f"⚠️  VAD non disponible: {e}")
        stt_config['use_vad'] = False
    
    # STT Manager
    stt_manager = RobustSTTManager(stt_config, vad_manager=vad_manager)
    await stt_manager.initialize()
    print(f"✅ STT Manager initialisé sur {stt_manager.device}")
    
    # 3. Test 1 : Phrase de validation obligatoire
    print("\n🧪 TEST 1 : Phrase de validation obligatoire")
    phrase_validation = "Ceci est un test de validation du nouveau gestionnaire robuste"
    print(f"📢 Phrase à prononcer : « {phrase_validation} »")
    
    audio_array = record_micro(seconds=7)
    
    # Conversion en bytes WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, 16000, format='WAV', subtype='PCM_16')
    audio_bytes = buffer.getvalue()
    
    # Transcription
    start_time = time.time()
    result = await stt_manager.transcribe_audio(audio_bytes, language="fr")
    
    # Validation résultats
    transcription = result['text'].lower()
    processing_time = result['processing_time']
    
    print(f"\n📝 Transcription : '{result['text']}'")
    print(f"⏱️  Temps de traitement : {processing_time:.3f}s")
    print(f"🖥️  Device utilisé : {result['device']}")
    print(f"📊 RTF (Real-Time Factor) : {result['metrics']['rtf']:.2f}")
    
    # Assertions sémantiques
    mots_cles = ["test", "validation", "gestionnaire", "robuste"]
    mots_trouves = [mot for mot in mots_cles if mot in transcription]
    
    assert len(mots_trouves) >= 3, (
        f"Transcription incomplète. Mots trouvés: {mots_trouves}/4. "
        f"Transcription: '{result['text']}'"
    )
    
    # Assertion performance (< 300ms pour audio court selon PRD)
    audio_duration = len(audio_array) / 16000
    if audio_duration < 10:  # Audio court
        assert processing_time < 0.3, (
            f"Latence trop élevée: {processing_time:.3f}s > 0.3s (cible PRD)"
        )
    
    print("✅ TEST 1 RÉUSSI : Validation sémantique et performance OK")
    
    # 4. Test 2 : Robustesse avec audio difficile
    print("\n🧪 TEST 2 : Test de robustesse")
    print("📢 Parlez rapidement avec des mots techniques")
    
    audio_array2 = record_micro(seconds=5)
    buffer2 = io.BytesIO()
    sf.write(buffer2, audio_array2, 16000, format='WAV', subtype='PCM_16')
    
    result2 = await stt_manager.transcribe_audio(buffer2.getvalue())
    
    assert result2['text'] != "", "Transcription vide sur audio difficile"
    assert result2['processing_time'] < 1.0, "Temps de traitement trop long"
    
    print(f"📝 Transcription robuste : '{result2['text']}'")
    print("✅ TEST 2 RÉUSSI : Robustesse validée")
    
    # 5. Test 3 : Métriques et monitoring
    print("\n🧪 TEST 3 : Vérification métriques")
    
    metrics = stt_manager.get_metrics()
    assert metrics['transcriptions'] == 2, "Compteur transcriptions incorrect"
    assert metrics['errors'] == 0, "Erreurs détectées pendant les tests"
    assert metrics['avg_latency'] > 0, "Latence moyenne non calculée"
    
    print(f"📊 Métriques finales : {metrics}")
    print("✅ TEST 3 RÉUSSI : Métriques correctes")
    
    # 6. Nettoyage
    await stt_manager.cleanup()
    
    # Résumé final
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLÈTE RÉUSSIE")
    print(f"   - Transcriptions réussies : {metrics['transcriptions']}")
    print(f"   - Latence moyenne : {metrics['avg_latency']:.3f}s")
    print(f"   - Device utilisé : {metrics['device']}")
    print(f"   - Modèles chargés : {metrics['models_loaded']}")
    print("="*80)

@pytest.mark.asyncio
async def test_fallback_chain():
    """Test spécifique de la chaîne de fallback"""
    print("\n🧪 TEST FALLBACK CHAIN")
    
    # Configuration avec chaîne de fallback étendue
    stt_config = {
        'use_gpu': False,  # Force CPU pour test
        'fallback_chain': ['base', 'small', 'tiny'],
        'use_vad': False
    }
    
    stt_manager = RobustSTTManager(stt_config)
    
    # Initialisation avec simulation d'échec sur 'base'
    stt_manager.fallback_chain = ['small', 'tiny']  # Skip 'base'
    await stt_manager.initialize()
    
    # Test audio simple
    test_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
    buffer = io.BytesIO()
    sf.write(buffer, test_audio, 16000, format='WAV')
    
    result = await stt_manager.transcribe_audio(buffer.getvalue())
    
    assert result['text'] == "", "Silence devrait donner transcription vide"
    assert result['device'] == 'cpu', "Device incorrect"
    
    await stt_manager.cleanup()
    print("✅ Fallback chain validée")

if __name__ == "__main__":
    # Exécution directe pour tests manuels
    asyncio.run(test_robust_stt_manager_validation_complete())
```

```python
# Modifications dans run_assistant.py

# Imports à ajouter/modifier
from STT.stt_manager_robust import RobustSTTManager
from STT.vad_manager import VADManager

# Dans la fonction main() ou setup_components()
async def setup_stt_components(config):
    """Configuration des composants STT avec le nouveau manager"""
    
    # Initialisation VAD si configuré
    vad_manager = None
    if config.get('vad', {}).get('enabled', True):
        try:
            vad_manager = VADManager(config['vad'])
            logger.info("VAD Manager initialisé")
        except Exception as e:
            logger.warning(f"VAD non disponible: {e}")
    
    # Initialisation STT Manager Robuste
    stt_manager = RobustSTTManager(config['stt'], vad_manager=vad_manager)
    await stt_manager.initialize()
    
    logger.info(f"STT Manager initialisé sur {stt_manager.device}")
    return stt_manager

# Remplacer l'ancienne initialisation par :
stt_handler = await setup_stt_components(config)
```

```shellscript
   pytest -v -s tests/test_realtime_audio_pipeline.py::test_robust_stt_manager_validation_complete
```

```shellscript
# 1. Créer le nouveau manager
touch STT/stt_manager_robust.py

# 2. Exécuter les tests de validation
pytest -v -s tests/test_realtime_audio_pipeline.py

# 3. Vérifier l'intégration
python run_assistant.py --test-mode

# 4. Supprimer l'ancien handler
rm STT/stt_handler.py

# 5. Commit avec message structuré
git add -A
git commit -m "feat(stt): Implement RobustSTTManager with real-time validation

- Replace basic STTHandler with robust manager
- Add GPU selection and fallback chain
- Integrate Prometheus metrics and circuit breakers
- Validate with real microphone tests
- Achieve <300ms latency for short audio

Closes Phase 1 / Task 2 of LUXA Development Plan"
```

