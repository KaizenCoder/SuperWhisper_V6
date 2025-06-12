"""
Handler TTS MVP P0 utilisant Microsoft Hortense (voix française Windows native)
"""

import os
import time
import tempfile
import wave
import numpy as np
import sounddevice as sd
import win32com.client

class TTSHandlerMVP:
    """Handler TTS MVP utilisant voix française Windows native"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        self.use_gpu = config.get('use_gpu', False)  # N/A pour SAPI
        
        # Initialiser SAPI Windows
        self._init_sapi()
        
    def _init_sapi(self):
        """Initialiser SAPI avec voix française"""
        print("🔄 Initialisation TTS MVP (voix française Windows)...")
        
        try:
            self.sapi = win32com.client.Dispatch("SAPI.SpVoice")
            voices = self.sapi.GetVoices()
            
            # Chercher voix française
            french_voice = None
            print(f"🔍 Recherche voix française parmi {voices.Count} voix...")
            
            for i in range(voices.Count):
                voice = voices.Item(i)
                name = voice.GetDescription()
                
                if any(keyword in name.lower() for keyword in ['french', 'français', 'hortense']):
                    french_voice = voice
                    print(f"✅ Voix française MVP trouvée: {name}")
                    break
            
            if french_voice:
                self.sapi.Voice = french_voice
                self.voice_name = french_voice.GetDescription()
            else:
                print("⚠️ Voix française non trouvée, utilisation voix par défaut")
                self.voice_name = "Voix par défaut"
            
            # Configuration optimale pour LUXA
            self.sapi.Rate = 1    # Vitesse légèrement accélérée
            self.sapi.Volume = 100  # Volume maximum
            
            print(f"✅ TTS MVP initialisé avec: {self.voice_name}")
            self.sapi_available = True
            
        except Exception as e:
            print(f"❌ Erreur initialisation SAPI: {e}")
            self.sapi_available = False
            
    def synthesize_to_file(self, text, output_path):
        """Synthèse vers fichier WAV"""
        if not self.sapi_available:
            return False
            
        try:
            # Utiliser win32com pour synthèse fichier
            file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
            file_stream.Open(output_path, 3)  # SSFMCreateForWrite
            
            # Rediriger sortie vers fichier
            original_output = self.sapi.AudioOutputStream
            self.sapi.AudioOutputStream = file_stream
            
            # Synthèse
            self.sapi.Speak(text)
            
            # Restaurer sortie
            file_stream.Close()
            self.sapi.AudioOutputStream = original_output
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur synthèse fichier MVP: {e}")
            return False
            
    def load_wav_file(self, file_path):
        """Charger fichier WAV et convertir en numpy array"""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Convertir en numpy array
                if sample_width == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32767.0
                else:
                    # Autres formats
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32767.0
                
                # Gérer stéréo → mono
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                return audio_data, sample_rate
                
        except Exception as e:
            print(f"❌ Erreur lecture WAV MVP: {e}")
            return None, None
            
    def synthesize(self, text):
        """Synthèse vocale MVP française"""
        print(f"🇫🇷 Synthèse TTS MVP (Microsoft Hortense)")
        print(f"   Texte: '{text}'")
        
        if not self.sapi_available:
            print("❌ SAPI MVP non disponible")
            return np.array([], dtype=np.int16)
        
        try:
            start_time = time.time()
            
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Synthèse vers fichier
            success = self.synthesize_to_file(text, tmp_path)
            
            if success and os.path.exists(tmp_path):
                # Charger audio
                audio_data, wav_sample_rate = self.load_wav_file(tmp_path)
                
                if audio_data is not None:
                    synth_time = time.time() - start_time
                    chars_per_sec = len(text) / synth_time if synth_time > 0 else 0
                    
                    print(f"   🎵 Audio MVP généré: {len(audio_data)} échantillons")
                    print(f"   📊 Sample rate: {wav_sample_rate}Hz")
                    print(f"   ⚡ Performance MVP: {chars_per_sec:.0f} car/s")
                    print(f"   🔍 Range audio: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
                    
                    # Vérifier qualité
                    amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                    if amplitude > 0.01:
                        print(f"   ✅ Audio MVP français valide (amplitude: {amplitude:.3f})")
                    else:
                        print(f"   ⚠️ Audio MVP français faible (amplitude: {amplitude:.3f})")
                    
                    # Conversion pour lecture
                    audio_int16 = (audio_data * 32767).clip(-32767, 32767).astype(np.int16)
                    
                    # Nettoyage
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    return audio_int16
                    
            # Nettoyage en cas d'erreur
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return np.array([], dtype=np.int16)
            
        except Exception as e:
            print(f"❌ Erreur synthèse MVP: {e}")
            import traceback
            traceback.print_exc()
            return np.array([], dtype=np.int16)
    
    def speak(self, text):
        """Synthèse et lecture audio MVP français"""
        audio_data = self.synthesize(text)
        
        if len(audio_data) > 0:
            print(f"   🔊 Lecture audio MVP français...")
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()
            print(f"   ✅ Lecture MVP française terminée")
        else:
            print(f"   ❌ Pas d'audio MVP français à lire")
            
        return audio_data
        
    def get_info(self):
        """Informations sur le handler MVP"""
        return {
            'name': 'TTS MVP Français',
            'engine': 'Microsoft SAPI',
            'voice': getattr(self, 'voice_name', 'Non configurée'),
            'language': 'Français',
            'quality': 'Production ready',
            'performance': 'Rapide',
            'gpu_support': False
        } 