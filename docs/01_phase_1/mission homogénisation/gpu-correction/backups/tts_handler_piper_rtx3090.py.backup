# TTS/tts_handler_piper_rtx3090.py
import os
import tempfile
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import onnxruntime

# Configuration RTX 3090 OBLIGATOIRE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # RTX 3090 24GB
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

class TTSHandlerPiperRTX3090:
    def __init__(self, config):
        self.config = config
        
        # Chemin vers le modèle français local  
        self.model_path = config.get('model_path', 'D:\\TTS_Voices\\piper\\fr_FR-siwis-medium.onnx')
        self.use_gpu = config.get('use_gpu', True)
        self.sample_rate = config.get('sample_rate', 22050)
        
        # Paramètres de synthèse
        self.noise_scale = config.get('noise_scale', 0.667)
        self.noise_scale_w = config.get('noise_scale_w', 0.8)
        self.length_scale = config.get('length_scale', 1.0)
        
        # Charger le modèle ONNX avec RTX 3090
        self._load_model()
        
        # Cache de phonémisation
        self._phoneme_cache = {}
        
    def _load_model(self):
        """Charger le modèle ONNX avec providers RTX 3090"""
        print(f"🔄 Chargement du modèle Piper RTX 3090: {self.model_path}")
        
        # Providers optimisés pour RTX 3090
        providers = []
        if self.use_gpu:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,  # GPU 0 dans le contexte CUDA_VISIBLE_DEVICES='1'
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB sur RTX 3090
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
        providers.append('CPUExecutionProvider')
        
        try:
            # Créer la session ONNX avec RTX 3090
            self.session = onnxruntime.InferenceSession(
                self.model_path, 
                providers=providers
            )
            
            # Vérifier le provider actuel
            current_providers = self.session.get_providers()
            print(f"🚀 Providers ONNX RTX 3090: {current_providers}")
            
            if 'CUDAExecutionProvider' in current_providers:
                print("✅ RTX 3090 CUDA activé pour TTS Piper")
            else:
                print("⚠️ Fallback CPU (DLLs CUDA manquantes)")
                
        except Exception as e:
            print(f"❌ Erreur chargement modèle RTX 3090: {e}")
            raise
            
        # Obtenir les métadonnées du modèle
        self.model_inputs = [inp.name for inp in self.session.get_inputs()]
        self.model_outputs = [out.name for out in self.session.get_outputs()]
        
        print(f"📝 Inputs modèle: {self.model_inputs}")
        print(f"📤 Outputs modèle: {self.model_outputs}")
        
    def text_to_phonemes(self, text):
        """Conversion texte vers phonèmes (simplifiée pour test)"""
        if text in self._phoneme_cache:
            return self._phoneme_cache[text]
            
        # Conversion basique caractère → ID
        # Note: Un vrai système utiliserait espeak ou phonemizer
        char_to_id = {
            ' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
            'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
            't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 27, '!': 28, '?': 29,
            ',': 30, ';': 31, ':': 32, 'à': 33, 'é': 34, 'è': 35, 'ê': 36, 'ë': 37, 'ç': 38, 'ù': 39, 'û': 40
        }
        
        phoneme_ids = []
        for char in text.lower():
            if char in char_to_id:
                phoneme_ids.append(char_to_id[char])
            else:
                phoneme_ids.append(0)  # Espace par défaut
                
        self._phoneme_cache[text] = phoneme_ids
        return phoneme_ids
        
    def synthesize(self, text):
        """Synthèse vocale avec RTX 3090"""
        print(f"🔊 Synthèse vocale Piper RTX 3090 en cours...")
        print(f"   Texte: '{text}'")
        
        try:
            # 1. Conversion texte → phonèmes
            phoneme_ids = self.text_to_phonemes(text)
            print(f"   📝 Phonèmes générés: {len(phoneme_ids)} IDs")
            
            # 2. Préparation des inputs pour ONNX
            input_ids = np.array([phoneme_ids], dtype=np.int64)
            input_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
            
            # Scales pour contrôler la synthèse
            scales = np.array([self.noise_scale, self.length_scale, self.noise_scale_w], dtype=np.float32)
            
            print(f"   🔍 Input shape: {input_ids.shape}, lengths: {input_lengths}")
            
            # 3. Inférence RTX 3090
            onnx_inputs = {
                'input': input_ids,
                'input_lengths': input_lengths, 
                'scales': scales
            }
            
            # Vérification des inputs
            print(f"   🔍 Model inputs: {self.model_inputs}")
            print(f"   🔍 Our inputs: {list(onnx_inputs.keys())}")
            
            # Exécution sur RTX 3090
            outputs = self.session.run(None, onnx_inputs)
            audio_data = outputs[0]
            
            print(f"   🔍 Raw audio shape: {audio_data.shape}")
            
            # 4. Post-processing audio
            audio_data = np.squeeze(audio_data)
            print(f"   🔍 Squeezed audio shape: {audio_data.shape}")
            print(f"   🎵 Audio brut généré: {audio_data.shape} échantillons")
            
            # Normalisation et conversion
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            print(f"   🔍 Audio type: {type(audio_data)}, dtype: {audio_data.dtype}")
            print(f"   🔍 Final audio: {audio_data.shape}, range: [{int(audio_data.min())}, {int(audio_data.max())}]")
            
            # Conversion pour lecture audio
            audio_int16 = (audio_data * 32767).astype(np.int16)
            print(f"   🔧 Audio converti: {len(audio_int16)} échantillons int16")
            
            return audio_int16
            
        except Exception as e:
            print(f"❌ Erreur synthèse RTX 3090: {e}")
            import traceback
            traceback.print_exc()
            return np.array([], dtype=np.int16)
    
    def speak(self, text):
        """Synthèse et lecture audio avec RTX 3090"""
        print(f"🔊 Synthèse vocale Piper RTX 3090 en cours...")
        
        audio_data = self.synthesize(text)
        if len(audio_data) > 0:
            print(f"   🔊 Lecture audio directe...")
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()  # Attendre la fin de la lecture
            print(f"   ✅ Synthèse terminée - {len(audio_data)} échantillons")
        else:
            print(f"   ❌ Aucun audio généré")
            
        print(f"🔊 Fin de la synthèse.")
        return audio_data
        
    def save_audio(self, text, filepath):
        """Synthèse et sauvegarde audio RTX 3090"""
        audio_data = self.synthesize(text)
        if len(audio_data) > 0:
            sf.write(filepath, audio_data, self.sample_rate)
            print(f"🎵 Audio sauvé: {filepath}")
        return len(audio_data) > 0 