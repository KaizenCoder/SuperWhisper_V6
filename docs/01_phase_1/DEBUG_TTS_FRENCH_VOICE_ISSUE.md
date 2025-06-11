# DEBUG TTS FRENCH VOICE ISSUE - PROMPT POUR DÉVELOPPEUR O3

**Date**: 2025-06-10  
**Projet**: LUXA SuperWhisper V6  
**Problème**: Synthèse vocale TTS génère une voix non-française malgré l'utilisation du modèle `fr_FR-siwis-medium.onnx`

---

## 🎯 PROMPT POUR DÉVELOPPEUR O3

**Mission**: Diagnostiquer pourquoi le système TTS de LUXA génère une voix non-française alors qu'il utilise le modèle Piper `fr_FR-siwis-medium.onnx` qui devrait produire une voix française.

**Contexte critique**:
- Le système fonctionnait correctement dans le passé (voir journal de développement)
- Performance technique excellente (RTX 3090 actif, vitesse 1333 car/s)
- Problème de langue uniquement : voix générée n'est pas française
- Configuration semble correcte mais résultat audio incorrect

**Questions spécifiques**:
1. Pourquoi le modèle `fr_FR-siwis-medium.onnx` génère-t-il une voix non-française ?
2. Y a-t-il une différence entre les modèles dans `models/` vs `D:\TTS_Voices\piper\` ?
3. Le paramètre `--speaker 0` est-il correct pour ce modèle ?
4. Y a-t-il un problème de configuration ou de chemin ?
5. Comment valider qu'un modèle Piper est bien français ?

---

## 📊 SYMPTÔMES OBSERVÉS

### ✅ Performance Technique (OK)
```
🎮 GPU RTX 3090: ✅ Actif
⚡ Performance moyenne: 1333 caractères/s
🚀 RTF: ~0.03 (très rapide)
📝 Phonèmes: 259-393 IDs générés correctement
🎵 Audio généré: 147200-245504 échantillons
🔍 Range audio: [-0.416, 0.456] (normal)
```

### ❌ Problème de Langue (CRITIQUE)
```
🗣️ Test 1: "Bonjour ! Je suis LUXA..." → Voix non-française
🗣️ Test 2: "LUXA utilise un pipeline..." → Voix non-française  
🗣️ Test 3: "Dans un futur proche..." → Voix non-française
🗣️ Test 4: "Pour configurer votre..." → Voix non-française
🗣️ Test 5: "Vous savez, l'intelligence..." → Voix non-française

📊 Feedback utilisateur: 4/4 (Faible) sur tous les tests
💬 Commentaires: "voix pas en français", "régression"
```

---

## 🔧 CODES SOURCES PERTINENTS

### 1. Configuration Projet (`Config/mvp_settings.yaml`)
```yaml
# Config/mvp_settings.yaml
# Configuration minimale pour le MVP P0

stt:
  model_name: "openai/whisper-base"
  gpu_device: "cuda:0"

llm:
  model_path: "D:/modeles_llm/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_S.gguf"
  gpu_device_index: 0
  n_gpu_layers: -1

tts:
  # Configuration pour Piper-TTS local (100% offline, conforme LUXA)
  model_path: "models/fr_FR-siwis-medium.onnx"
  use_gpu: true
  sample_rate: 22050
```

### 2. Script de Test qui Échoue (`test_tts_long_feedback.py`)
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test TTS Piper avec textes longs pour feedback qualité vocale
Évaluation complète de la compréhensibilité et prosodie
"""

import sys
import os
import time

# Configuration RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def test_tts_feedback_long():
    """Test TTS avec textes longs pour feedback qualité"""
    
    print("🎤 TEST TTS PIPER - FEEDBACK QUALITÉ VOCALE")
    print("=" * 60)
    
    # Configuration PROBLÉMATIQUE - CHEMIN DIFFÉRENT DU PROJET
    config = {
        'model_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx',
        'config_path': 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json',
        'use_gpu': True
    }
    
    try:
        # Import et initialisation
        print("1. 🚀 Initialisation handler RTX 3090...")
        from TTS.tts_handler_piper_fixed import TTSHandlerPiperFixed
        
        start_time = time.time()
        handler = TTSHandlerPiperFixed(config)
        init_time = time.time() - start_time
        print(f"✅ Handler initialisé en {init_time:.2f}s")
        
        # Tests de qualité avec différents scénarios français
        test_scenarios = [
            {
                "name": "📖 PRÉSENTATION LUXA",
                "text": "Bonjour ! Je suis LUXA, votre assistant vocal intelligent..."
            },
            # ... autres scénarios
        ]
        
        # Chaque test génère une voix NON-FRANÇAISE
        for scenario in test_scenarios:
            audio_data = handler.synthesize(scenario['text'])
            handler.speak(scenario['text'])  # PROBLÈME ICI
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
```

### 3. TTSHandler Principal (`TTS/tts_handler_piper_fixed.py`)
```python
# TTS/tts_handler_piper_fixed.py
import subprocess
import json
import soundfile as sf
import sounddevice as sd
import numpy as np
from pathlib import Path
import tempfile
import os
import time

class TTSHandlerPiperFixed:
    def __init__(self, config):
        self.config = config
        self.model_path = config['model_path']
        self.use_gpu = config.get('use_gpu', False)
        self.sample_rate = 22050
        
        # Configuration Piper
        self.piper_executable = "piper/piper.exe"
        
        # Chargement configuration modèle
        config_path = config.get('config_path', f"{self.model_path}.json")
        
        print(f"📄 Chargement config: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.model_config = json.load(f)
        
        # Informations du modèle
        self.num_phonemes = len(self.model_config.get('phoneme_id_map', {}))
        self.sample_rate = self.model_config.get('sample_rate', 22050)
        
        print(f"✅ Config chargée: {self.num_phonemes} phonèmes")
        print(f"   Sample rate: {self.sample_rate}Hz")
        
        # Paramètres de synthèse depuis la config
        inference_config = self.model_config.get('inference', {})
        self.noise_scale = inference_config.get('noise_scale', 0.667)
        self.length_scale = inference_config.get('length_scale', 1.0)
        self.noise_w = inference_config.get('noise_w', 0.8)
        
        print(f"   Paramètres: noise={self.noise_scale}, length={self.length_scale}, noise_w={self.noise_w}")
        
        # Vérification du modèle
        print(f"🔄 Chargement modèle RTX 3090: {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")
        
        # Test de chargement avec onnxruntime
        import onnxruntime as ort
        
        # Configuration session ONNX
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers
        )
        
        print(f"🚀 Providers: {self.session.get_providers()}")
        print(f"✅ RTX 3090 CUDA activé" if 'CUDAExecutionProvider' in self.session.get_providers() else "⚠️ CPU seulement")

    def synthesize(self, text):
        """Synthèse vocale avec piper.exe CLI"""
        try:
            print(f"🔊 Synthèse Piper FIXÉE (RTX 3090)")
            print(f"   Texte: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            # Fichier temporaire pour l'output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            try:
                # Commande piper CLI AVEC TOUS LES PARAMÈTRES
                cmd = [
                    self.piper_executable,
                    '--model', self.model_path,
                    '--output_file', output_path,
                    '--speaker', '0',  # PARAMÈTRE CRITIQUE
                    '--noise_scale', str(self.noise_scale),
                    '--length_scale', str(self.length_scale), 
                    '--noise_w', str(self.noise_w)
                ]
                
                if self.use_gpu:
                    cmd.append('--use_gpu')
                
                # Exécution avec le texte en input
                process = subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=30,
                    encoding='utf-8'
                )
                
                if process.returncode != 0:
                    print(f"❌ Erreur piper: {process.stderr}")
                    return np.array([])
                
                # Lecture du fichier audio généré
                if Path(output_path).exists():
                    audio_data, sr = sf.read(output_path)
                    
                    # Conversion mono si nécessaire
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    print(f" 📝 Phonèmes: {len(text)} caractères convertis")
                    print(f"   🎵 Audio généré: {audio_data.shape} échantillons")
                    print(f"   🔍 Range audio: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
                    
                    return audio_data
                else:
                    print(f"❌ Fichier audio non généré: {output_path}")
                    return np.array([])
                    
            finally:
                # Nettoyage fichier temporaire
                if Path(output_path).exists():
                    os.unlink(output_path)
                    
        except Exception as e:
            print(f"❌ Erreur synthèse: {e}")
            return np.array([])

    def speak(self, text):
        """Synthèse et lecture directe"""
        audio_data = self.synthesize(text)
        
        if len(audio_data) > 0:
            print(f"🔊 Lecture audio...")
            
            try:
                # Lecture avec sounddevice
                sd.play(audio_data, samplerate=self.sample_rate)
                sd.wait()  # Attendre la fin de la lecture
                print(f"   ✅ Lecture terminée")
                
            except Exception as e:
                print(f"❌ Erreur lecture: {e}")
        else:
            print(f"❌ Pas d'audio à lire")
```

### 4. Script de Validation PowerShell (`validate_piper.ps1`)
```powershell
Write-Host "🧪 Validation de bas niveau pour piper.exe..."
$PiperExecutable = ".\piper\piper.exe"
$ModelPath = ".\models\fr_FR-siwis-medium.onnx"
$OutputFile = "validation_output.wav"
$TestText = "Si vous entendez cette phrase, la validation de base de Piper est réussie."

if (-not (Test-Path $PiperExecutable)) {
    Write-Error "❌ ERREUR: L'exécutable Piper n'a pas été trouvé à l'emplacement '$PiperExecutable'."
    exit 1
}
if (-not (Test-Path $ModelPath)) {
    Write-Error "❌ ERREUR: Le modèle '$ModelPath' n'a pas été trouvé."
    exit 1
}

Write-Host "✅ Prérequis validés."
Write-Host "🔊 Lancement de la synthèse..."
echo $TestText | & $PiperExecutable --model $ModelPath --output_file $OutputFile --speaker 0 --use_gpu
if (Test-Path $OutputFile) {
    Write-Host "✅ Fichier '$OutputFile' généré. Écoutez-le pour valider."
    Invoke-Item $OutputFile
} else {
    Write-Error "❌ La génération a échoué."
}
```

---

## 📋 HISTORIQUE DU PROBLÈME

### D'après le Journal de Développement

**Problème Initial Résolu** (Section "Résolution problème TTS Piper"):
```
- Problème root cause: Modèle fr_FR-upmc-medium défectueux/incompatible
- Solution: Modèle de remplacement fr_FR-siwis-medium.onnx depuis Hugging Face
- Tests réussis: "3 synthèses vocales parfaites avec audio output"
- Qualité confirmée: "qualité audio excellente, latence acceptable"
- Architecture finale: TTSHandler hybride CLI + Python parfaitement fonctionnel
```

**État Actuel** (Régression détectée):
```
- Performance technique: ✅ Excellente (1333 car/s, RTX 3090 actif)
- Qualité vocale: ❌ Voix non-française sur TOUS les tests
- Feedback utilisateur: 4/4 (Faible) avec commentaires "voix pas en français"
```

---

## 🔍 INCOHÉRENCES DÉTECTÉES

### 1. **Différence de Chemins Modèles**
- **Configuration projet**: `models/fr_FR-siwis-medium.onnx`
- **Test utilisé**: `D:\TTS_Voices\piper\fr_FR-siwis-medium.onnx`

### 2. **Régression vs Historique**
- **Journal**: "3 synthèses vocales parfaites avec audio output"
- **Actuel**: Voix non-française sur tous les tests

### 3. **Performance vs Qualité**
- **Technique**: Parfaite (GPU, vitesse, phonèmes)
- **Linguistique**: Complètement échouée (non-français)

---

## 🎯 STRUCTURE FICHIERS

```
C:\Dev\SuperWhisper_V6\
├── models/
│   └── fr_FR-siwis-medium.onnx          # Configuration projet
├── piper/
│   └── piper.exe                         # Exécutable CLI
├── TTS/
│   └── tts_handler_piper_fixed.py       # Handler principal
├── Config/
│   └── mvp_settings.yaml                # Config projet
└── test_tts_long_feedback.py            # Test qui échoue

D:\TTS_Voices\piper\
├── fr_FR-siwis-medium.onnx              # Modèle du test (suspect)
└── fr_FR-siwis-medium.onnx.json         # Config du test
```

---

## ❓ QUESTIONS SPÉCIFIQUES POUR O3

1. **Validation modèle**: Comment vérifier qu'un fichier `.onnx` Piper est bien français ?
2. **Paramètre speaker**: Le `--speaker 0` est-il correct pour `fr_FR-siwis-medium` ?
3. **Différence modèles**: Y a-t-il une différence entre les deux fichiers `fr_FR-siwis-medium.onnx` ?
4. **Configuration JSON**: Le fichier `.json` contient-il des paramètres de langue ?
5. **Debug piper**: Comment diagnostiquer pourquoi piper.exe génère la mauvaise langue ?
6. **Commande CLI**: La commande piper CLI utilisée est-elle correcte ?

---

## 📄 LOGS DE SORTIE DÉTAILLÉS

```
🎮 GPU RTX 3090: ✅ Actif
📄 Chargement config: D:\TTS_Voices\piper\fr_FR-siwis-medium.onnx.json
✅ Config chargée: 154 phonèmes
   Sample rate: 22050Hz
   Paramètres: noise=0.667, length=1, noise_w=0.8
🔄 Chargement modèle RTX 3090: D:\TTS_Voices\piper\fr_FR-siwis-medium.onnx
🚀 Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
✅ RTX 3090 CUDA activé

🔊 Synthèse Piper FIXÉE (RTX 3090)
   Texte: 'Bonjour ! Je suis LUXA, votre assistant vocal intelligent...'
 📝 Phonèmes: 259 IDs - [1, 15, 27, 26, 22, 27, 33, 30, 3, 4]...
   🎵 Audio généré: (147200,) échantillons
   🔍 Range audio: [-0.416, 0.456]
   ⚡ Synthèse: 0.35s (745 car/s)
   🎵 Durée audio: 6.7s
   🚀 RTF: 0.052
   🔊 LECTURE EN COURS...

RÉSULTAT: Voix générée NON-FRANÇAISE (feedback utilisateur)
```

---

## 🎯 OBJECTIF FINAL

**Identifier pourquoi** un modèle `fr_FR-siwis-medium.onnx` qui devrait générer une voix française produit une voix dans une autre langue, et **fournir une solution** pour corriger ce problème dans le contexte du projet LUXA.

La performance technique est parfaite, le problème est uniquement linguistique/qualité vocale. 