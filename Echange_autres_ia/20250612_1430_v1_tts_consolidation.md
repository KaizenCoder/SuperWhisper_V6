# 🎯 CONSULTATION AVIS ALTERNATIF - CONSOLIDATION TTS SUPERWHISPER V6

**Timestamp :** 2025-06-12 14:30  
**Version :** v1  
**Phase :** TTS Consolidation  
**Objectif :** Solliciter avis alternatif sur stratégie consolidation TTS

---

## 📋 PARTIE 1 : CONTEXTE COMPLET

### 🎯 **VISION GLOBALE**

**SuperWhisper V6 (LUXA)** est un **assistant vocal intelligent 100% local** avec une architecture modulaire STT → LLM → TTS. L'objectif est de créer une expérience voix-à-voix naturelle sans dépendance cloud, optimisée pour la performance et la confidentialité.

### 🏗️ **ARCHITECTURE TECHNIQUE**

#### **Pipeline Vocal Principal**
```
🎤 MICROPHONE → STT (Whisper) → LLM (Llama) → TTS (Piper) → 🔊 SPEAKERS
                    ↓               ↓              ↓
                VAD Manager    Context Manager   Audio Output
```

#### **Modules Core Identifiés**
- **STT/** : Speech-to-Text avec Whisper (insanely-fast-whisper)
- **LLM/** : Language Model avec llama-cpp-python  
- **TTS/** : Text-to-Speech avec Piper (15+ handlers TTS !)
- **Orchestrator/** : Coordination pipeline + fallback management

### 🚨 **CONFIGURATION GPU CRITIQUE**

**Configuration Dual-GPU Obligatoire :**
- **RTX 5060 (8GB) sur CUDA:0** ❌ **INTERDITE D'UTILISATION**
- **RTX 3090 (24GB) sur CUDA:1** ✅ **SEULE GPU AUTORISÉE**

⚠️ **Point d'attention majeur** : Les règles GPU sont clairement définies mais doivent être respectées absolument.

**Configuration Système Complète :**
- RAM : 64GB DDR4-4800
- CPU : Intel Core Ultra 7 265K (20 threads)
- Stockage : 50GB+ modèles IA locaux

### 📊 **ÉTAT ACTUEL DU PROJET**

#### **✅ Modules Fonctionnels (6/18)**
- `memory_leak_v4.py` ✅
- `TTS/tts_handler_coqui.py` ✅
- `TTS/tts_handler_piper_native.py` ✅  
- `LLM/llm_manager_enhanced.py` ✅
- `STT/stt_manager_robust.py` ✅
- `Orchestrator/master_handler_robust.py` ✅

#### **❌ Défis Identifiés**
- **12/18 modules non-fonctionnels** selon le rapport de validation
- **Fragmentation TTS** : 15+ handlers TTS différents (besoin de consolidation)
- **Dépendances manquantes** : Plusieurs modules ont des problèmes d'import
- **Scripts principaux manquants** : `superwhisper_v6.py` introuvable

### 🎯 **OBJECTIFS PERFORMANCE**

#### **Cibles Techniques :**
- **Latence Pipeline** : < 1.2s total (STT <300ms + LLM <500ms + TTS <120ms)
- **Précision STT** : > 95% français  
- **Qualité TTS** : MOS > 4.0
- **Disponibilité** : 99.9%

### 📋 **PHASES DE DÉVELOPPEMENT**

#### **Phase 1 - Corrections Critiques (EN COURS)**
- ✅ Correction import bloquant (TERMINÉE)
- 🔄 Implémentation RobustSTTManager (3 jours)
- 🔄 Consolidation UnifiedTTSManager (3 jours) 
- 🔄 EnhancedLLMManager avec contexte (4 jours)

#### **Phase 2 - Extensions Intelligentes (PLANIFIÉE)**
- Hot-swap multi-modèles VRAM (5 jours)
- Interface sélection microphone (4 jours)
- Optimisations GPU SuperWhisper2 (5 jours)

#### **Phase 3 - Tests & Finalisation (PLANIFIÉE)**
- Suite benchmarks performance (4 jours)  
- Tests de charge résistance (3 jours)
- Dashboard monitoring Grafana (3 jours)

### 🔧 **STACK TECHNIQUE**

#### **Technologies Core :**
- **Python 3.12** avec async/await
- **STT** : insanely-fast-whisper + Whisper OpenAI
- **LLM** : llama-cpp-python + modèles GGUF
- **TTS** : Piper.exe (CLI) + multiples backends
- **GPU** : CUDA 11.8+ optimisation NVIDIA

#### **Gestion Modèles :**
- **Stockage** : ~10-15GB total requis
- **Cache** : models/ avec sous-dossiers spécialisés
- **VRAM** : Répartition optimale dual-GPU

### 📁 Écosystème et Structure
```
SuperWhisper_V6/
├── STT/ (Speech-to-Text - Whisper optimisé)
├── LLM/ (Language Model - Llama local)  
├── TTS/ (Text-to-Speech - 15 handlers !) ← PROBLÈME
├── Orchestrator/ (Coordination pipeline)
├── Config/ (Configuration système)
├── docs/ (Documentation complète)
└── tests/ (Validation continue)
```

### 📈 **MÉTRIQUES ACTUELLES**

| Métrique | Valeur | Statut |
|----------|--------|--------|
| **Modules Fonctionnels** | 6/18 (33%) | 🔄 En développement |
| **Phase Actuelle** | Phase 1 - 25% | 🔄 Corrections critiques |
| **TTS** | ✅ Finalisé | ✅ 3 tests réussis |
| **GPU Config** | ⚠️ Critique | ⚠️ RTX 3090 uniquement |
| **Documentation** | 95% | ✅ Excellente |

### 🚨 **POINTS D'ATTENTION CRITIQUE**

#### **Risques Majeurs :**
1. **Fragmentation TTS** : 15+ handlers à consolider
2. **Compliance GPU** : Risque d'utilisation accidentielle RTX 5060
3. **Dépendances** : Modules avec imports cassés
4. **Integration** : Pipeline end-to-end non validé

#### **Opportunités :**
1. **Architecture solide** : Structure modulaire bien pensée
2. **Performance TTS** : <120ms déjà atteint
3. **Documentation** : Très complète avec Task Master
4. **Stack mature** : Technologies éprouvées

### 🎖️ **RECOMMANDATIONS STRATÉGIQUES**

#### **Priorité Immédiate :**
1. **Consolider TTS** : UnifiedTTSManager pour remplacer les 15+ handlers
2. **Valider GPU** : S'assurer de la conformité RTX 3090 exclusive
3. **Fix Dependencies** : Résoudre les imports cassés
4. **Test Pipeline** : Validation end-to-end STT→LLM→TTS

#### **Moyen Terme :**
1. **Performance** : Optimisation latence < 1.2s
2. **Robustesse** : Fallbacks + circuit breakers  
3. **Monitoring** : Dashboard temps réel
4. **Tests** : Couverture >80%

### 🎯 **CONCLUSION CONTEXTE**

**LUXA** est un projet **ambitieux et bien structuré** avec une vision claire d'assistant vocal 100% local. L'architecture modulaire est solide, mais le projet nécessite une **consolidation technique urgente** pour passer de 33% à 100% de modules fonctionnels.

**Phase critique** : Les 3 prochaines semaines détermineront le succès du projet avec la finalisation des managers robustes et l'intégration pipeline complète.

**Forces** : Documentation excellente, architecture claire, performance TTS validée  
**Défis** : Fragmentation technique, conformité GPU, intégration modules

Le projet est **viable et prometteur** mais nécessite une **execution disciplinée** sur les corrections critiques Phase 1.

### 🔍 Handler TTS Fonctionnel Validé

**Script complet : `TTS/tts_handler.py`**
```python
# TTS/tts_handler.py
"""
TTSHandler utilisant l'exécutable piper en ligne de commande
Solution de contournement pour éviter les problèmes avec piper-phonemize
"""

import json
import subprocess
import tempfile
import wave
from pathlib import Path
import numpy as np
import sounddevice as sd

class TTSHandler:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.speaker_map = {}
        self.piper_executable = None
        
        print("🔊 Initialisation du moteur TTS Piper (avec gestion multi-locuteurs)...")
        
        model_p = Path(self.model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Fichier modèle .onnx non trouvé : {self.model_path}")
        
        config_p = Path(f"{self.model_path}.json")
        if not config_p.exists():
            raise FileNotFoundError(f"Fichier de configuration .json non trouvé : {config_p}")

        # Charger la carte des locuteurs depuis le fichier JSON
        self._load_speaker_map(config_p)
        
        # Chercher l'exécutable piper
        self._find_piper_executable()
        
        if self.piper_executable:
            print("✅ Moteur TTS Piper chargé avec succès.")
        else:
            raise FileNotFoundError("Exécutable piper non trouvé")

    def _find_piper_executable(self):
        """Cherche l'exécutable piper dans différents emplacements."""
        possible_paths = [
            "piper/piper.exe",  # Répertoire local (Windows)
            "piper.exe",  # Dans le PATH (Windows)
            "bin/piper.exe",  # Répertoire bin (Windows)
            "./piper.exe",  # Répertoire courant (Windows)
            "piper/piper",  # Répertoire local (Linux/macOS)
            "piper",  # Dans le PATH (Linux/macOS)
            "./piper",  # Répertoire courant (Linux/macOS)
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    self.piper_executable = path
                    return
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue

    def _load_speaker_map(self, config_path: Path):
        """Charge la carte des locuteurs depuis le fichier de configuration."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # Vérifier le nombre de locuteurs
            num_speakers = config_data.get("num_speakers", 1)
            
            if num_speakers > 1:
                # La structure peut varier, nous cherchons 'speaker_id_map'
                if "speaker_id_map" in config_data and config_data["speaker_id_map"]:
                    # La carte est souvent imbriquée, ex: {'vits': {'speaker_name': 0}}
                    # On prend la première carte non vide trouvée.
                    for key, value in config_data["speaker_id_map"].items():
                        if value:
                            self.speaker_map = value
                            break

                if self.speaker_map:
                    print("🗣️ Locuteurs disponibles détectés dans le modèle :")
                    for name, sid in self.speaker_map.items():
                        print(f"  - {name} (ID: {sid})")
                else:
                    print(f"⚠️ Modèle déclaré multi-locuteurs ({num_speakers} locuteurs) mais speaker_id_map vide.")
                    print("   Utilisation du locuteur par défaut (ID: 0)")
            else:
                print("ℹ️ Modèle mono-locuteur détecté (num_speakers = 1).")
                print("   Utilisation du locuteur par défaut (ID: 0)")

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture des locuteurs : {e}")

    def speak(self, text: str):
        """Synthétise le texte en parole en utilisant l'exécutable piper avec gestion des locuteurs."""
        if not text:
            print("⚠️ Texte vide, aucune synthèse à faire.")
            return

        if not self.piper_executable:
            print("❌ Exécutable Piper non disponible")
            return

        # Déterminer le speaker_id à utiliser
        # Pour ce MVP, nous utiliserons l'ID 0 par défaut
        speaker_id = 0
        if self.speaker_map:
            # Si nous avons une carte des locuteurs, utiliser le premier disponible
            speaker_id = next(iter(self.speaker_map.values()))
            print(f"🎭 Utilisation du locuteur avec l'ID : {speaker_id}")
        else:
            print("🎭 Utilisation du locuteur par défaut (ID: 0)")
        
        print(f"🎵 Synthèse Piper pour : '{text}'")
        
        try:
            # Créer un fichier temporaire pour la sortie
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Construire la commande piper
            cmd = [
                self.piper_executable,
                "--model", str(self.model_path),
                "--output_file", tmp_path,
                "--speaker", str(speaker_id)  # Toujours inclure le speaker_id
            ]
            
            # Exécuter piper avec le texte en entrée
            result = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Lire et jouer le fichier généré
                if Path(tmp_path).exists():
                    self._play_wav_file(tmp_path)
                    print("✅ Synthèse Piper terminée avec succès.")
                else:
                    print("❌ Fichier de sortie non généré")
            else:
                print(f"❌ Erreur piper (code {result.returncode}):")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout lors de l'exécution de piper")
        except Exception as e:
            print(f"❌ Erreur durant la synthèse Piper : {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Nettoyer le fichier temporaire
            try:
                if 'tmp_path' in locals():
                    Path(tmp_path).unlink(missing_ok=True)
            except:
                pass

    def _play_wav_file(self, file_path):
        """Joue un fichier WAV."""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Convertir en numpy array
                if sample_width == 1:
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif sample_width == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32767.0
                else:
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                    audio_data = audio_data.astype(np.float32) / 2147483647.0
                
                # Gérer stéréo → mono
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                # Jouer l'audio
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()
                
        except Exception as e:
            print(f"❌ Erreur lecture WAV: {e}")
```

### 🧪 Validation Tests Réussis
```python
# Extrait test_tts_handler.py - Tests 3/3 réussis
✅ Test 1/3: "Bonjour, je suis LUXA, votre assistant vocal intelligent."
✅ Test 2/3: "Test de synthèse vocale avec le modèle français."  
✅ Test 3/3: "La synthèse fonctionne parfaitement!"
```

### 🔍 Problème Identifié : Fragmentation TTS
**15 handlers TTS détectés :**
```
HANDLERS PIPER (11 fichiers):
├─ tts_handler_piper_native.py      ← Candidat principal (défaillant)
├─ tts_handler_piper_rtx3090.py     ← Optimisé GPU (défaillant)
├─ tts_handler_piper_simple.py      ← Basique (non testé)
├─ tts_handler_piper_french.py      ← Français (non testé)
├─ tts_handler_piper_original.py    ← Legacy
├─ tts_handler_piper_direct.py      ← Legacy
├─ tts_handler_piper_espeak.py      ← Legacy
├─ tts_handler_piper_fixed.py       ← Legacy
├─ tts_handler_piper_cli.py         ← Legacy
├─ tts_handler_piper.py             ← Legacy
└─ tts_handler.py                   ← FONCTIONNEL ✅

HANDLERS AUTRES (4 fichiers):
├─ tts_handler_sapi_french.py       ← SAPI Windows (audio généré ✅)
├─ tts_handler_coqui.py             ← GPU alternatif
├─ tts_handler_fallback.py          ← Emergency (interface manquante)
└─ tts_handler_mvp.py               ← Basique MVP
```

### 🎯 Tests Validation Handlers (Résultats Factuels)
**Test des 4 candidats initiaux :**
- **piper_rtx3090** : ❌ Erreur modèle ONNX manquant
- **piper_native** : ❌ Erreur modèle manquant
- **sapi_french** : ✅ Audio généré (102876 échantillons, 22050Hz)
- **fallback** : ❌ Interface méthode synthèse manquante

**Handlers réellement fonctionnels identifiés :**
1. ✅ `TTS/tts_handler.py` (piper.exe CLI) - **3 tests vocaux réussis**
2. ✅ `TTS/tts_handler_sapi_french.py` - **Audio généré validé**

### 📈 Historique Développement TTS
1. **Problème initial** : Modèle `fr_FR-upmc-medium` défectueux
2. **Solution adoptée** : Modèle `fr_FR-siwis-medium` + piper.exe CLI
3. **Architecture finale** : subprocess + cleanup automatique + multi-locuteurs
4. **Validation** : 3 synthèses vocales parfaites avec audio output
5. **Performance** : <1s synthèse, qualité excellente

### 🎯 But Mission Consolidation
**Objectif :** Simplifier l'architecture TTS
- **Avant :** 15 handlers fragmentés et redondants
- **Après :** 2-4 handlers robustes avec fallback automatique
- **Bénéfices :** Maintenabilité, robustesse, clarté architecture

---

## ❓ QUESTIONS POUR AVIS ALTERNATIF

### 🎯 **Question Principale :**
**Quelle est votre approche recommandée pour consolider efficacement 15 handlers TTS fragmentés en une architecture unifiée robuste, en tenant compte des contraintes GPU RTX 3090 exclusive et des objectifs de performance <120ms ?**

### 🔍 **Questions Spécifiques :**

#### **1. Architecture & Design Pattern :**
- **Manager Pattern vs Factory Pattern** : Quel pattern recommandez-vous pour UnifiedTTSManager ?
- **Fallback Chain** : Comment structurer la chaîne de fallback (Piper → SAPI → Emergency) ?
- **Interface Unifiée** : Quelle signature d'API pour `synthesize(text, voice_config)` ?

#### **2. Gestion Performance & GPU :**
- **CUDA Memory Management** : Comment optimiser l'utilisation VRAM RTX 3090 pour TTS ?
- **Async/Await** : Architecture async recommandée pour pipeline non-bloquant ?
- **Caching Strategy** : Cache audio généré vs régénération à la demande ?

#### **3. Robustesse & Monitoring :**
- **Circuit Breaker Pattern** : Implémentation pour handlers défaillants ?
- **Métriques Temps Réel** : Quelles métriques TTS critiques monitorer ?
- **Error Recovery** : Stratégie de récupération automatique en cas d'échec ?

#### **4. Validation & Tests :**
- **Test Strategy** : Comment valider qualité audio et latence <120ms ?
- **Regression Testing** : Approche pour éviter régressions lors consolidation ?
- **Load Testing** : Tests de charge pour usage intensif ?

#### **5. Migration & Déploiement :**
- **Migration Path** : Stratégie pour migrer de 15 handlers → UnifiedTTSManager ?
- **Backward Compatibility** : Maintenir compatibilité pendant transition ?
- **Rollback Strategy** : Plan B si consolidation échoue ?

### 🚨 **Contraintes Critiques à Respecter :**
- ✅ **GPU RTX 3090 exclusive** (CUDA:1 uniquement)
- ✅ **Performance <120ms** par synthèse
- ✅ **Handler Piper fonctionnel** à préserver
- ✅ **Architecture async/await** obligatoire
- ✅ **Fallback robuste** requis

### 🎖️ **Critères d'Évaluation Réponse :**
1. **Faisabilité technique** (complexité implémentation)
2. **Performance** (respect objectifs latence)
3. **Robustesse** (gestion erreurs + fallbacks)
4. **Maintenabilité** (code clean + documentation)
5. **Testabilité** (validation automatisée)

---

## 📋 PARTIE 2 : PROMPT D'EXÉCUTION

### 🎯 Mission Consolidation TTS SuperWhisper V6

**Objectif :** Consolider 15 handlers TTS fragmentés → 2 handlers fonctionnels avec fallback

### 🔍 Compréhension Factuelle Requise
Avant toute action, confirmer compréhension :
1. **Handler principal fonctionnel** : `TTS/tts_handler.py` (piper.exe CLI)
2. **Handler fallback fonctionnel** : `TTS/tts_handler_sapi_french.py` (SAPI Windows)
3. **Handlers à archiver** : 13 fichiers redondants/défaillants
4. **Voix utilisée** : `fr_FR-siwis-medium.onnx` (modèle Piper)

### 🛠️ Template de Consolidation Obligatoire
```python
class UnifiedTTSManager:
    def __init__(self, config: dict):
        # Validation GPU RTX 3090 obligatoire
        self._validate_rtx3090_exclusive()
        
        # Configuration backends fonctionnels uniquement
        self.backends = {
            'piper_cli': {
                'handler_class': TTSHandler,
                'config': config.get('piper', {}),
                'priority': 1,
                'description': 'Principal - Piper.exe CLI fr_FR-siwis-medium'
            },
            'sapi_french': {
                'handler_class': TTSHandlerSapiFrench,
                'config': config.get('sapi', {}), 
                'priority': 2,
                'description': 'Fallback - SAPI Windows français natif'
            }
        }
        
        self.active_handlers = {}
        self._initialize_handlers()
    
    def synthesize(self, text: str, prefer_backend: str = None):
        """Synthèse TTS avec fallback automatique"""
        # Ordre : piper_cli → sapi_french
        # Retour : {'success': bool, 'backend_used': str, 'latency_ms': float}
```

### ⚠️ Contraintes Critiques
1. **GPU Configuration** : `CUDA_VISIBLE_DEVICES='1'` + `CUDA_DEVICE_ORDER='PCI_BUS_ID'`
2. **Préservation handlers fonctionnels** : NE PAS modifier tts_handler.py
3. **Archivage sécurisé** : Timestamp + documentation rollback
4. **Tests obligatoires** : Validation TTS avant/après consolidation

### 🎯 Critères de Succès
- ✅ **15 handlers → 2 handlers** (réduction 87%)
- ✅ **Fallback automatique** piper_cli → sapi_french
- ✅ **Performance préservée** : <1s latence principale
- ✅ **Interface unifiée** : `.synthesize()` standardisée
- ✅ **Documentation rollback** : Instructions restauration

---

## 📋 PARTIE 3 : PRD CONSOLIDATION TTS

### 📊 Contexte Business
**Problème :** 15 handlers TTS fragmentés causent complexité maintenance et risques instabilité
**Solution :** Architecture TTS unifiée avec 2 backends robustes et fallback automatique
**Impact :** Réduction 87% complexité, amélioration maintenabilité, robustesse accrue

### 🎯 Objectifs Quantifiables
1. **Consolidation structurelle** : 15 → 2 handlers (réduction 87%)
2. **Performance préservée** : Latence <1s maintenue
3. **Robustesse améliorée** : Fallback automatique fonctionnel
4. **Interface standardisée** : API unifiée `.synthesize()`
5. **Documentation complète** : Guide rollback + standards futurs

### 🏗️ Spécifications Techniques

#### Configuration GPU Obligatoire
```python
# Configuration RTX 3090 exclusive
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre physique
# Résultat : cuda:0 = RTX 3090
```

#### Architecture UnifiedTTSManager
```python
class UnifiedTTSManager:
    backends = {
        'piper_cli': TTSHandler,              # Principal (fonctionnel)
        'sapi_french': TTSHandlerSapiFrench   # Fallback (fonctionnel)
    }
    
    def synthesize(text) -> dict:
        # Fallback automatique : piper_cli → sapi_french
        # Retour : {'success': bool, 'backend_used': str, 'latency_ms': float}
```

### 📋 Exigences Fonctionnelles
1. **Synthèse TTS** : Interface `.synthesize()` unifiée
2. **Fallback automatique** : Basculement transparent si échec principal
3. **Validation GPU** : RTX 3090 exclusive obligatoire
4. **Gestion erreurs** : Logs détaillés + recovery gracieux
5. **Configuration** : Support backends via config YAML

### 📋 Exigences Non-Fonctionnelles
1. **Performance** : Latence <1s (piper_cli), <2s (sapi_french)
2. **Disponibilité** : 99.9% via fallback automatique
3. **Maintenabilité** : Code modulaire + documentation complète
4. **Robustesse** : Gestion pannes + recovery automatique
5. **Sécurité** : Validation inputs + sanitization

### 📁 Fichiers Impactés
```
À CONSERVER (2 fichiers):
✅ TTS/tts_handler.py                    # Principal fonctionnel
✅ TTS/tts_handler_sapi_french.py        # Fallback fonctionnel

À CRÉER (1 fichier):
📝 TTS/tts_manager_unified.py            # Manager consolidé

À ARCHIVER (13 fichiers):
🗂️ TTS/tts_handler_piper_*.py (10 fichiers redondants)
🗂️ TTS/tts_handler_coqui.py
🗂️ TTS/tts_handler_mvp.py  
🗂️ TTS/tts_handler_fallback.py
```

### 🧪 Stratégie de Test
1. **Tests unitaires** : Chaque backend individuellement
2. **Tests intégration** : Fallback automatique
3. **Tests performance** : Latence comparative avant/après
4. **Tests robustesse** : Simulation pannes + recovery
5. **Tests régression** : Validation fonctionnalités préservées

### ⚠️ Risques et Mitigation
| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Régression fonctionnelle | CRITIQUE | Faible | Tests exhaustifs + rollback Git |
| Performance dégradée | ÉLEVÉ | Faible | Benchmarks comparatifs continus |
| Handlers manquants | MOYEN | Moyen | Archivage sécurisé + documentation |

### 🎯 Critères d'Acceptation
- ✅ UnifiedTTSManager fonctionnel avec 2 backends
- ✅ Fallback automatique piper_cli → sapi_french
- ✅ Performance ≥ baseline (latence <1s principal)
- ✅ 13 handlers archivés avec documentation rollback
- ✅ Tests automatisés 100% passants
- ✅ Interface `.synthesize()` standardisée

---

## 📋 PARTIE 4 : PLAN DE DÉVELOPPEMENT

### 🕒 Planning Détaillé (2 jours)

#### **JOUR 1 - Audit et Archivage**

**09h00-10h00 : Validation Handlers Fonctionnels (1h)**
- ✅ Confirmer `TTS/tts_handler.py` fonctionnel (tests 3/3 réussis)
- ✅ Valider `TTS/tts_handler_sapi_french.py` (audio généré validé)
- 📋 Documenter interfaces et configurations existantes

**10h00-14h00 : Archivage Sécurisé (4h)**
```bash
# Création archive timestampée
mkdir TTS/legacy_handlers_20250612/

# Documentation archivage
cat > TTS/legacy_handlers_20250612/README.md << EOF
# Archive Handlers TTS - 12 juin 2025
Consolidation 15→2 handlers suite tests validation.
Handlers archivés car non-fonctionnels/redondants.

## Rollback
mv legacy_handlers_20250612/*.py ../
EOF

# Migration 13 handlers
mv TTS/tts_handler_piper_native.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_rtx3090.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_piper_simple.py TTS/legacy_handlers_20250612/
# ... [10 autres handlers piper]
mv TTS/tts_handler_coqui.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_mvp.py TTS/legacy_handlers_20250612/
mv TTS/tts_handler_fallback.py TTS/legacy_handlers_20250612/
```

**14h00-17h00 : Préparation Tests (3h)**
- 🧪 Tests baseline handlers conservés
- 📊 Benchmarks performance référence
- 📋 Spécifications UnifiedTTSManager

#### **JOUR 2 - Implémentation et Tests**

**09h00-12h00 : Implémentation UnifiedTTSManager (3h)**
```python
# TTS/tts_manager_unified.py
class UnifiedTTSManager:
    def __init__(self, config: dict):
        # Validation GPU RTX 3090
        self._validate_rtx3090_exclusive()
        
        # Backends fonctionnels
        self.backends = {
            'piper_cli': {
                'handler': TTSHandler,
                'config': config.get('tts', {}),
                'priority': 1
            },
            'sapi_french': {
                'handler': TTSHandlerSapiFrench,
                'config': config.get('sapi', {}),
                'priority': 2
            }
        }
        
        self._initialize_backends()
    
    def synthesize(self, text: str):
        """Synthèse avec fallback automatique"""
        for backend_name in ['piper_cli', 'sapi_french']:
            try:
                start_time = time.time()
                handler = self.active_handlers[backend_name]
                handler.speak(text)
                
                latency_ms = (time.time() - start_time) * 1000
                return {
                    'success': True,
                    'backend_used': backend_name,
                    'latency_ms': latency_ms
                }
            except Exception as e:
                print(f"Backend {backend_name} failed: {e}")
                continue
        
        raise RuntimeError("Tous backends TTS échoué")
```

**12h00-15h00 : Tests Complets (3h)**
```python
# tests/test_unified_tts_manager.py
def test_primary_backend():
    """Test backend principal (Piper CLI)"""
    manager = UnifiedTTSManager(config)
    result = manager.synthesize("Test Piper")
    
    assert result['success'] == True
    assert result['backend_used'] == 'piper_cli'
    assert result['latency_ms'] < 1000

def test_fallback_automatic():
    """Test fallback automatique"""
    # Simulation panne Piper
    with patch.object(TTSHandler, 'speak', side_effect=Exception("Piper failed")):
        manager = UnifiedTTSManager(config)
        result = manager.synthesize("Test fallback")
        
        assert result['success'] == True
        assert result['backend_used'] == 'sapi_french'

def test_performance_regression():
    """Test absence régression performance"""
    # Benchmark avant/après consolidation
    assert new_latency <= baseline_latency * 1.02
```

**15h00-16h00 : Intégration (1h)**
```python
# Mise à jour run_assistant.py
from TTS.tts_manager_unified import UnifiedTTSManager

# Remplacement handler TTS
tts_manager = UnifiedTTSManager(config)
```

**16h00-17h00 : Documentation (1h)**
- 📋 Rapport consolidation détaillé
- 📚 Guide utilisation UnifiedTTSManager  
- 🔄 Instructions rollback si nécessaire

### 🎯 Livrables Attendus
1. ✅ **UnifiedTTSManager fonctionnel** (2 backends)
2. ✅ **13 handlers archivés** avec documentation
3. ✅ **Tests automatisés** 100% passants
4. ✅ **Performance préservée** (latence <1s)
5. ✅ **Intégration run_assistant.py** réussie
6. ✅ **Documentation complète** rollback + utilisation

### 📊 Métriques de Succès
- **Réduction complexité** : 15 → 2 handlers (87%)
- **Latence principale** : <1s (Piper CLI)
- **Latence fallback** : <2s (SAPI French)  
- **Taux réussite tests** : 100%
- **Couverture documentation** : 100%

---

## 🤔 QUESTION POUR AVIS ALTERNATIF

**Contexte :** Nous avons identifié 2 handlers TTS fonctionnels sur 15, et planifions une consolidation vers un UnifiedTTSManager.

**Questions spécifiques :**
1. **Stratégie consolidation** : L'approche 15→2 handlers est-elle optimale ?
2. **Architecture fallback** : Le fallback automatique Piper→SAPI est-il suffisant ?
3. **Gestion des 13 handlers défaillants** : Archivage vs tentative réparation ?
4. **Interface unifiée** : Le design UnifiedTTSManager répond-il aux besoins ?
5. **Risques non identifiés** : Quels points critiques manquons-nous ?

**Alternatives considérer :**
- Réparation des handlers défaillants au lieu d'archivage ?
- Architecture différente pour le fallback ?
- Stratégie de migration progressive ?
- Tests supplémentaires requis ?

**Merci pour votre avis éclairé sur cette stratégie de consolidation TTS !** 



___________________
réponse suite à la consultation:
Absolument. Voici la réponse complète et consolidée, intégrant tous les éléments discutés et finalisés, prête à servir de document de référence pour l'implémentation.

🚀 Dossier de Consolidation TTS - SuperWhisper V6 (Version Finale)
Ce document constitue le livrable final de la consultation et du design de l'architecture pour la consolidation du sous-système Text-to-Speech (TTS).

1. Le Code Exhaustif
Voici l'ensemble du code de l'architecture cible, structuré en fichiers prêts à être intégrés au projet.

config/tts.yaml
Fichier de configuration centralisé, découplant la logique du comportement.

YAML

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
    model_path: "models/TTS/fr_FR-siwis-medium.onnx"
    model_config_path: "models/TTS/fr_FR-siwis-medium.onnx.json"
    device: "cuda:0"  # Pointera vers RTX 3090 après CUDA_VISIBLE_DEVICES
    speaker_id: 0
    target_latency_ms: 120

  # Priorité 2: Fallback fonctionnel (CPU)
  piper_cli:
    enabled: true
    model_path: "models/TTS/fr_FR-siwis-medium.onnx"
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
TTS/tts_manager.py
Cœur de l'architecture : le manager unifié, les handlers et les composants de robustesse.

Python

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

2. Le Prompt d'Exécution
Mission : Finaliser la Consolidation du Sous-Système TTS de SuperWhisper V6

Objectif Principal : Remplacer les 15 handlers TTS fragmentés par l'architecture UnifiedTTSManager finalisée, en respectant le code, la configuration et le plan de développement ci-joints.

Critères de Succès Impératifs :

Performance : Atteindre une latence de synthèse strictement inférieure à 120ms pour le handler principal (piper_native sur GPU).
Robustesse : La chaîne de fallback à 4 niveaux (Piper Natif → Piper CLI → SAPI → Urgence Silencieux) doit être 100% fonctionnelle et testée.
Propreté du Code : Les 13 handlers obsolètes doivent être archivés de manière sécurisée. Le code du projet doit uniquement référencer le UnifiedTTSManager.
Configuration : Toute la configuration doit être gérée via le fichier config/tts.yaml, sans aucune valeur codée en dur.
Livrable Attendu : Une Pull Request contenant l'implémentation complète et passant 100% des tests de validation (performance, fallback, régression).

3. Le PRD (Product Requirements Document) - Version Finale
(Basé sur le document original 20250612_1430_v1_tts_consolidation.md et mis à jour avec les décisions finales)

Section	Description
Problème	La maintenance de 15 handlers TTS fragmentés est insoutenable et crée des risques d'instabilité. Les performances ne sont pas garanties.
Solution	Mettre en place une architecture UnifiedTTSManager centralisée, configurable et robuste, avec une chaîne de fallback à 4 niveaux pour garantir la performance et la disponibilité.
Impact Business	Réduction drastique de la complexité technique (-87% de fichiers), amélioration de la maintenabilité, et garantie d'une expérience utilisateur fluide et réactive (<120ms), essentielle pour un assistant vocal.

Exporter vers Sheets
Exigences Fonctionnelles (FR)
FR1 : Synthèse Unifiée : L'application doit exposer une unique méthode asynchrone UnifiedTTSManager.synthesize() pour toutes les opérations TTS.
FR2 : Fallback Automatique : En cas d'échec du handler principal (piper_native), le système doit basculer de manière transparente et ordonnée sur piper_cli, puis sapi_french, puis silent_emergency.
FR3 : Isolation des Pannes : Un handler défaillant doit être automatiquement mis à l'écart pendant une durée configurable (Circuit Breaker).
FR4 : Gestion de Cache : Les synthèses fréquentes doivent être mises en cache pour une réponse quasi instantanée (<5ms).
FR5 : Configuration Externe : Tous les paramètres (chemins, timeouts, features) doivent être gérés via le fichier config/tts.yaml.
Exigences Non-Fonctionnelles (NFR)
NFR1 : Performance : La latence de synthèse du handler piper_native doit être < 120ms en P95 (95ème percentile).
NFR2 : Disponibilité : Le service TTS doit avoir une disponibilité de 99.9% grâce à la chaîne de fallback.
NFR3 : Utilisation GPU : Le sous-système TTS ne doit pas utiliser plus de 10% de la VRAM de la RTX 3090. La RTX 5060 ne doit jamais être utilisée.
NFR4 : Qualité Audio : La qualité audio générée par les backends Piper doit atteindre un score MOS > 4.0.
NFR5 : Maintenabilité : Le code doit être entièrement typé, documenté (docstrings) et couvert par des tests.
4. Le Plan de Développement - Ajusté
(Basé sur la version finale consolidée)

Phase 0 : Préparation (0.5 jour)
Créer une branche de feature : feature/tts-consolidation.
Créer un tag Git de sauvegarde : git tag pre-tts-consolidation.
Créer le script de rollback : scripts/rollback_tts.sh qui effectue un git checkout pre-tts-consolidation et nettoie les nouveaux fichiers.
Archiver les 13 handlers obsolètes dans TTS/legacy_handlers_20250612/.
Phase 1 : Réparation Piper Natif & Base du Manager (2 jours)
Objectif : Atteindre <120ms.
Analyser les dépendances et erreurs des handlers Piper défaillants (ex: tts_handler_piper_native.py).
Implémenter le PiperNativeHandler fonctionnel, en s'assurant de l'intégration correcte avec le GPU.
Créer le squelette du UnifiedTTSManager et intégrer ce premier handler.
Valider par des benchmarks que la latence est bien sous la barre des 120ms. C'est un point de blocage pour la suite.
Phase 2 : Implémentation Complète du Manager (2 jours)
Intégrer les 3 autres handlers (PiperCli, Sapi, SilentEmergency) dans le UnifiedTTSManager.
Implémenter et tester les composants CircuitBreaker et TTSCache.
Connecter l'ensemble au TTSConfigLoader pour une gestion via config/tts.yaml.
Rédiger les tests unitaires pour chaque composant et les tests d'intégration pour la chaîne de fallback (en simulant des pannes).
Phase 3 : Déploiement & Validation (1 jour)
Intégrer le UnifiedTTSManager dans le pipeline global de l'application (dans l'Orchestrator).
Utiliser un feature flag (lu depuis la config) pour activer le nouveau manager.
Exécuter la suite de tests de validation complète (performance, qualité, robustesse) pour comparer avant/après.
Monitorer intensivement les métriques (latence, taux d'erreur, utilisation du fallback).
Une fois validé, supprimer le feature flag et nettoyer l'ancien code d'appel.
Fusionner la Pull Request.
5. Réponse aux Questions Initiales - Version Finale
🎯 Question Principale : Quelle est votre approche recommandée pour consolider efficacement 15 handlers TTS fragmentés en une architecture unifiée robuste, en tenant compte des contraintes GPU RTX 3090 exclusive et des objectifs de performance <120ms ?
L'approche recommandée est de construire un UnifiedTTSManager basé sur les design patterns Manager, Strategy et Chain of Responsibility. Ce manager orchestrera quatre "stratégies" de synthèse (handlers) dans une chaîne de fallback robuste : Piper Natif (GPU) → Piper CLI (CPU) → SAPI (CPU) → Urgence Silencieux. La priorité absolue est de réparer et d'optimiser le handler Piper Natif pour qu'il s'exécute sur la RTX 3090 et atteigne la cible de <120ms, seule garantie de performance. La robustesse est assurée par des Circuit Breakers et la flexibilité par une configuration externe via un fichier YAML.

🔍 Questions Spécifiques :
1. Architecture & Design Pattern :

Manager Pattern vs Factory Pattern : Un Manager Pattern est recommandé, car il gère le cycle de vie, l'orchestration et la surveillance des handlers, et pas seulement leur création. Il implémente en interne le Strategy Pattern, où chaque handler est une stratégie interchangeable.
Fallback Chain : La chaîne doit être Piper Natif (GPU) → Piper CLI (CPU) → SAPI (CPU) → Silent Emergency Handler. Cette structure priorise la performance, puis la qualité, puis la disponibilité, et enfin la survie du système.
Interface Unifiée : Une interface asynchrone unique async def synthesize(text: str, voice: Optional[str] = None, speed: Optional[float] = None, reuse_cache: bool = True) -> TTSResult. Elle abstrait toute la complexité et retourne un dataclass TTSResult standardisé.
2. Gestion Performance & GPU :

CUDA Memory Management : Valider l'usage exclusif de la RTX 3090 via CUDA_VISIBLE_DEVICES='1', et allouer une fraction fixe et limitée de sa VRAM (ex: 10%) au processus TTS via torch.cuda.set_per_process_memory_fraction(), laissant le reste au LLM.
Async/Await : Le manager doit être entièrement asynchrone. Les appels à des librairies bloquantes (comme Piper natif) doivent être exécutés dans des threads séparés via asyncio.to_thread. Les appels à des processus externes (comme Piper CLI) doivent utiliser asyncio.create_subprocess_exec.
Caching Strategy : Un cache en mémoire (LRU) pour les phrases les plus fréquentes, avec un TTL (Time-to-Live) pour éviter les données obsolètes. La régénération à la demande reste la norme pour les textes uniques issus du LLM.
3. Robustesse & Monitoring :

Circuit Breaker Pattern : Chaque handler doit être encapsulé dans une instance de CircuitBreaker. Après N échecs consécutifs, le circuit s'ouvre et le handler est mis à l'écart pendant une période de réinitialisation, évitant de surcharger un service défaillant.
Métriques Temps Réel : Les métriques critiques à exporter (ex: vers Prometheus) sont : la latence de synthèse par backend (tts_synthesis_duration_seconds), le nombre d'appels par backend et par statut (tts_synthesis_total), le nombre de déclenchements de fallback (tts_fallback_triggered_total), et l'état des circuit breakers (tts_backend_health).
Error Recovery : La stratégie de récupération principale est la chaîne de fallback automatique. Elle est complétée par les Circuit Breakers. L'ultime niveau de récupération est le SilentEmergencyHandler qui empêche le crash de l'application.
4. Validation & Tests :

Test Strategy : Une approche à plusieurs niveaux : tests unitaires pour chaque handler et composant ; tests d'intégration pour valider la logique de fallback (en simulant des pannes) ; benchmarks de performance pour valider la latence <120ms ; et tests de qualité audio (MOS > 4.0).
Regression Testing : Une suite de tests automatisée, lancée en intégration continue (CI), qui compare la sortie audio (ex: par hash) et la performance par rapport à une baseline pour chaque changement de code.
Load Testing : Utiliser des outils pour simuler un grand nombre de requêtes concurrentes afin de mesurer la latence au 95ème/99ème percentile et de vérifier la stabilité du système et de l'utilisation mémoire sous stress.
5. Migration & Déploiement :

Migration Path : Un plan de migration progressif en 4 phases (Préparation, Réparation, Consolidation, Déploiement) comme détaillé ci-dessus. L'utilisation d'un feature flag (ex: --enable-unified-tts) est recommandée pour basculer le trafic en toute sécurité.
Backward Compatibility : La compatibilité est gérée au niveau de l'orchestrateur, qui appellera le nouveau manager. Il n'y a pas besoin de maintenir les anciennes interfaces une fois la migration validée.
Rollback Strategy : Un plan à deux niveaux : 1. Un script de rollback automatisé (rollback_tts.sh) qui utilise un tag Git pour restaurer instantanément l'état précédent. 2. Le feature flag qui permet de désactiver le nouveau système en production sans redéployer.