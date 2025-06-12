# TTS/tts_handler_piper_direct.py
import os
import tempfile
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import onnxruntime

class TTSHandlerPiperDirect:
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
        
        # Charger le modèle ONNX
        try:
            print(f"🔄 Chargement du modèle Piper: {self.model_path}")
            
            # Configuration ONNX Runtime
            sess_options = onnxruntime.SessionOptions()
            
            if self.use_gpu:
                # Essayer d'utiliser CUDA si disponible
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.model = onnxruntime.InferenceSession(
                str(self.model_path), 
                sess_options=sess_options,
                providers=providers
            )
            
            # Vérifier le provider utilisé
            provider_used = self.model.get_providers()[0]
            print(f"✅ TTS Handler Piper Direct initialisé - Modèle: {self.model_path}")
            print(f"🚀 Provider ONNX: {provider_used}")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle Piper: {e}")
            raise

    def text_to_phonemes(self, text):
        """
        Conversion améliorée du texte en phonèmes.
        Génère un mapping plus réaliste pour tester la synthèse.
        """
        # Mappage plus étendu avec pad et tokens spéciaux
        chars = "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? àâäéèêëïîôöùûüÿç"
        char_to_id = {char: i for i, char in enumerate(chars)}
        
        # Remplacer les caractères non reconnus par des espaces
        text = ''.join(char if char in char_to_id else ' ' for char in text)
        
        # Convertir en IDs avec un mapping plus réaliste
        phoneme_ids = []
        for char in text:
            if char in char_to_id:
                phoneme_ids.append(char_to_id[char])
            else:
                phoneme_ids.append(0)  # padding
        
        # Ensure minimum length and add padding
        if len(phoneme_ids) < 10:
            phoneme_ids.extend([0] * (10 - len(phoneme_ids)))
            
        return phoneme_ids

    def synthesize_audio(self, phoneme_ids):
        """Synthèse audio à partir des IDs de phonèmes"""
        
        # Préparer les entrées pour le modèle ONNX
        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)
        scales = np.array(
            [self.noise_scale, self.length_scale, self.noise_scale_w],
            dtype=np.float32,
        )
        
        # Debug: afficher les dimensions
        print(f"   🔍 Debug - text shape: {text.shape}, lengths: {text_lengths}")
        
        # Préparer les inputs pour le modèle
        inputs = {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales
        }
        
        # Ajouter sid si nécessaire (vérifions les inputs du modèle)
        input_names = [inp.name for inp in self.model.get_inputs()]
        if "sid" in input_names:
            inputs["sid"] = np.array([0], dtype=np.int64)  # speaker ID par défaut
        
        print(f"   🔍 Debug - Model inputs: {input_names}")
        print(f"   🔍 Debug - Our inputs: {list(inputs.keys())}")
        
        # Inférence
        audio = self.model.run(None, inputs)[0]
        print(f"   🔍 Debug - Raw audio shape: {audio.shape}")
        
        # Squeeze les dimensions inutiles
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        print(f"   🔍 Debug - Squeezed audio shape: {audio.shape}")
        
        # Vérifier que nous avons bien de l'audio
        if audio.size == 0:
            raise ValueError("Aucun audio généré par le modèle")
        
        return audio

    def audio_float_to_int16(self, audio):
        """Convertir l'audio float32 en int16"""
        # Vérifier le type et la forme
        print(f"   🔍 Debug - Audio type: {type(audio)}, dtype: {audio.dtype}")
        
        # S'assurer que l'audio est en float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normaliser et convertir
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        print(f"   🔍 Debug - Final audio: {audio_int16.shape}, range: [{audio_int16.min()}, {audio_int16.max()}]")
        
        return audio_int16

    def speak(self, text):
        """Synthétise et joue le texte avec Piper (100% local)."""
        print("🔊 Synthèse vocale Piper Direct en cours...")
        print(f"   Texte: '{text}'")
        
        try:
            # Étape 1: Conversion texte → phonèmes
            phoneme_ids = self.text_to_phonemes(text)
            print(f"   📝 Phonèmes générés: {len(phoneme_ids)} IDs")
            
            # Étape 2: Synthèse audio
            audio_raw = self.synthesize_audio(phoneme_ids)
            print(f"   🎵 Audio brut généré: {audio_raw.shape} échantillons")
            
            # Étape 3: Conversion du format
            audio = self.audio_float_to_int16(audio_raw)
            print(f"   🔧 Audio converti: {len(audio)} échantillons int16")
            
            # Étape 4: Sauvegarde et lecture directe
            if len(audio) > 0:
                # Lecture directe avec sounddevice (évite le problème de fichier)
                print(f"   🔊 Lecture audio directe...")
                audio_float = audio.astype(np.float32) / 32767.0  # Conversion pour sounddevice
                sd.play(audio_float, self.sample_rate)
                sd.wait()  # Attendre la fin de la lecture
                
                print(f"   ✅ Synthèse terminée - {len(audio)} échantillons")
            else:
                print("   ⚠️ Aucun audio généré")
                
        except Exception as e:
            print(f"❌ Erreur synthèse Piper Direct: {e}")
            import traceback
            traceback.print_exc()
            
        print("🔊 Fin de la synthèse.")

    def test_synthesis(self):
        """Test rapide de synthèse."""
        test_text = "Bonjour, je suis LUXA, votre assistant vocal local."
        print(f"🧪 Test de synthèse: '{test_text}'")
        self.speak(test_text) 