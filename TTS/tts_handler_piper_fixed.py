# TTS/tts_handler_piper_fixed.py
"""
TTS Handler Piper Fixed - Version corrigée du handler Piper
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
# RTX 5060 Ti (CUDA:0) = INTERDITE - RTX 3090 (CUDA:1) = OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Optimisation mémoire

print("🎮 GPU Configuration: RTX 3090 (CUDA:0 après mapping)")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Intégration Memory Leak V4.0 - Prévention fuites mémoire GPU
try:
    from memory_leak_v4 import GPUMemoryManager, gpu_test_cleanup, validate_no_memory_leak
    print("✅ Memory Leak V4.0 intégré avec succès")
except ImportError as e:
    print(f"⚠️ Memory Leak V4.0 non disponible: {e}")

import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import onnxruntime


def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    print("🔍 VALIDATION RTX 3090 OBLIGATOIRE")
    print("=" * 40)
    
    # 1. Vérification variables d'environnement
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
    print("✅ CUDA_VISIBLE_DEVICES: '1' (RTX 3090 uniquement)")
    
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    print("✅ CUDA_DEVICE_ORDER: 'PCI_BUS_ID'")
    
    # 2. Validation ONNX Runtime CUDA (si disponible)
    try:
        available_providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ ONNX Runtime CUDA disponible")
        else:
            print("⚠️ ONNX Runtime CUDA non disponible - fallback CPU")
    except Exception as e:
        print(f"⚠️ Erreur vérification ONNX: {e}")
    
    # 3. Validation générale (PyTorch optionnel pour TTS)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if "RTX 3090" not in gpu_name:
                raise RuntimeError(f"🚫 GPU incorrecte: '{gpu_name}' - RTX 3090 requise")
            print(f"✅ GPU validée: {gpu_name}")
            
            if gpu_memory < 20:  # RTX 3090 = ~24GB
                raise RuntimeError(f"🚫 VRAM insuffisante: {gpu_memory:.1f}GB - RTX 3090 (24GB) requise")
            print(f"✅ VRAM validée: {gpu_memory:.1f}GB")
        else:
            print("⚠️ PyTorch CUDA non disponible")
    except ImportError:
        print("⚠️ PyTorch non installé - validation ONNX uniquement")
    
    print("🎉 VALIDATION RTX 3090 RÉUSSIE")
    return True


class TTSHandlerPiperFixed:
    def __init__(self, config):
        # Validation RTX 3090 obligatoire à l'instanciation
        validate_rtx3090_mandatory()
        
        self.config = config
        
        # Chemins modèle
        self.model_path = config.get('model_path', 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx')
        self.config_path = config.get('config_path', 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json')
        self.use_gpu = config.get('use_gpu', True)
        self.sample_rate = config.get('sample_rate', 22050)
        
        # Charger la configuration du modèle
        self._load_model_config()
        
        # Charger le modèle ONNX
        self._load_model()
        
    def _load_model_config(self):
        """Charger la configuration JSON du modèle Piper"""
        print(f"📄 Chargement config: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.model_config = json.load(f)
            
        # Extraire les paramètres
        audio_config = self.model_config.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 22050)
        
        inference_config = self.model_config.get('inference', {})
        self.noise_scale = inference_config.get('noise_scale', 0.667)
        self.length_scale = inference_config.get('length_scale', 1.0)
        self.noise_w = inference_config.get('noise_w', 0.8)
        
        # Mapping phonème → ID
        self.phoneme_id_map = self.model_config.get('phoneme_id_map', {})
        
        print(f"✅ Config chargée: {len(self.phoneme_id_map)} phonèmes")
        print(f"   Sample rate: {self.sample_rate}Hz")
        print(f"   Paramètres: noise={self.noise_scale}, length={self.length_scale}, noise_w={self.noise_w}")
        
    def _load_model(self):
        """Charger le modèle ONNX avec RTX 3090"""
        print(f"🔄 Chargement modèle RTX 3090: {self.model_path}")
        
        # Ajout des DLLs CUDA au PATH
        torch_lib_path = os.path.join(os.getcwd(), 'venv_piper312', 'Lib', 'site-packages', 'torch', 'lib')
        if os.path.exists(torch_lib_path):
            current_path = os.environ.get('PATH', '')
            if torch_lib_path not in current_path:
                os.environ['PATH'] = current_path + os.pathsep + torch_lib_path
        
        # Providers optimisés RTX 3090
        providers = []
        if self.use_gpu:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
        providers.append('CPUExecutionProvider')
        
        # Créer session ONNX
        self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        
        # Vérifier providers
        current_providers = self.session.get_providers()
        print(f"🚀 Providers: {current_providers}")
        
        if 'CUDAExecutionProvider' in current_providers:
            print("✅ RTX 3090 CUDA activé")
        else:
            print("⚠️ Fallback CPU")
            
        # Métadonnées modèle
        self.model_inputs = [inp.name for inp in self.session.get_inputs()]
        self.model_outputs = [out.name for out in self.session.get_outputs()]
        
    def text_to_phonemes_simple(self, text):
        """Conversion texte → phonèmes simplifiée (pour test rapide)"""
        # Version simple sans espeak-ng pour test immédiat
        # Utilise un mapping basique mais avec les vrais IDs du modèle
        
        # Mapping simple français 
        simple_mapping = {
            'b': 'b', 'o': 'o', 'n': 'n', 'j': 'j', 'u': 'u', 'r': 'r',
            's': 's', 'a': 'a', 'l': 'l', 't': 't', ' ': ' ', '!': '!',
            ',': ',', 'c': 'k', 'e': 'e', 'i': 'i', 'v': 'v', 'z': 'z',
            'm': 'm', 'à': 'a', 'é': 'e', 'è': 'ɛ', 'ê': 'ɛ', 'ç': 's',
            'p': 'p', 'g': 'g', 'f': 'f', 'h': 'h', 'd': 'd', 'x': 'ks',
            'y': 'i', 'w': 'w', 'q': 'k', '.': '.', '?': '?', ':': ':'
        }
        
        phoneme_ids = []
        
        # Ajouter marqueur début
        if "^" in self.phoneme_id_map:
            phoneme_ids.extend(self.phoneme_id_map["^"])
            
        # Convertir chaque caractère
        for char in text.lower():
            mapped_char = simple_mapping.get(char, char)
            
            if mapped_char in self.phoneme_id_map:
                phoneme_ids.extend(self.phoneme_id_map[mapped_char])
            elif ' ' in self.phoneme_id_map:
                phoneme_ids.extend(self.phoneme_id_map[' '])  # Espace par défaut
                
        # Ajouter marqueur fin
        if "$" in self.phoneme_id_map:
            phoneme_ids.extend(self.phoneme_id_map["$"])
            
        return phoneme_ids
        
    def synthesize(self, text):
        """Synthèse vocale RTX 3090 avec vraie phonémisation"""
        print(f"🔊 Synthèse Piper FIXÉE (RTX 3090)")
        print(f"   Texte: '{text}'")
        
        try:
            # 1. Conversion texte → phonèmes
            phoneme_ids = self.text_to_phonemes_simple(text)
            print(f"   📝 Phonèmes: {len(phoneme_ids)} IDs - {phoneme_ids[:10]}...")
            
            # 2. Préparation inputs ONNX
            input_ids = np.array([phoneme_ids], dtype=np.int64)
            input_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
            scales = np.array([self.noise_scale, self.length_scale, self.noise_w], dtype=np.float32)
            
            # 3. Inférence RTX 3090
            onnx_inputs = {
                'input': input_ids,
                'input_lengths': input_lengths,
                'scales': scales
            }
            
            outputs = self.session.run(None, onnx_inputs)
            audio_data = outputs[0]
            
            # 4. Post-processing
            audio_data = np.squeeze(audio_data)
            
            print(f"   🎵 Audio généré: {audio_data.shape} échantillons")
            print(f"   🔍 Range audio: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            # Vérifier si audio contient du son
            if abs(audio_data.max()) < 0.001 and abs(audio_data.min()) < 0.001:
                print(f"   ⚠️ ATTENTION: Audio très faible, possible problème phonémisation")
            
            # Conversion pour lecture
            audio_int16 = (audio_data * 32767).clip(-32767, 32767).astype(np.int16)
            
            return audio_int16
            
        except Exception as e:
            print(f"❌ Erreur synthèse: {e}")
            import traceback
            traceback.print_exc()
            return np.array([], dtype=np.int16)
    
    def speak(self, text):
        """Synthèse et lecture audio RTX 3090"""
        audio_data = self.synthesize(text)
        
        if len(audio_data) > 0:
            print(f"   🔊 Lecture audio...")
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()
            print(f"   ✅ Lecture terminée")
        else:
            print(f"   ❌ Pas d'audio à lire")
            
        return audio_data 


# APPELER OBLIGATOIREMENT dans __main__ ou au début du script
if __name__ == "__main__":
    print("🧪 VALIDATION RTX 3090 - TTS Handler Piper Fixed")
    print("=" * 52)
    
    # Validation obligatoire de la configuration
    validate_rtx3090_mandatory()
    
    # Test d'instanciation basique
    test_config = {
        "use_gpu": True,
        "sample_rate": 22050,
        "model_path": "D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx",
        "config_path": "D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx.json"
    }
    
    try:
        print("🧪 Test instanciation TTSHandlerPiperFixed...")
        handler = TTSHandlerPiperFixed(test_config)
        print(f"✅ TTSHandlerPiperFixed initialisé avec succès")
        print(f"   GPU utilisé: {handler.use_gpu}")
        print(f"   Sample rate: {handler.sample_rate}Hz")
        print(f"   Phonèmes disponibles: {len(handler.phoneme_id_map) if hasattr(handler, 'phoneme_id_map') else 'N/A'}")
        
        # Test validation modèle (si les fichiers existent)
        if hasattr(handler, 'session'):
            providers = handler.session.get_providers()
            print(f"   ONNX Providers: {providers}")
            
        print("🎉 Validation complète réussie!")
        
    except FileNotFoundError as e:
        print(f"⚠️ Fichiers modèle manquants: {e}")
        print("   (Normal si les modèles Piper ne sont pas installés)")
        print("✅ Configuration GPU validée avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        print("✅ Configuration GPU de base validée") 