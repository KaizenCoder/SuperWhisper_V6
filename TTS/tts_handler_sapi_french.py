# TTS/tts_handler_sapi_french.py
import os
import numpy as np
import sounddevice as sd
import time
import tempfile
import wave

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import win32com.client
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

class TTSHandlerSapiFrench:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        self.voice_id = None
        
        # Initialiser l'engine TTS
        self._init_sapi_engine()
        
    def _init_sapi_engine(self):
        """Initialiser le moteur SAPI Windows avec voix française"""
        print(f"🔄 Initialisation SAPI Windows...")
        
        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                voices = self.engine.getProperty('voices')
                
                # Chercher voix française
                french_voice = None
                print(f"🔍 Recherche voix française parmi {len(voices)} voix...")
                
                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_id = voice.id.lower()
                    
                    print(f"   Voix: {voice.name} ({voice.id})")
                    
                    if any(keyword in voice_name for keyword in ['french', 'français', 'france', 'hortense', 'julie']):
                        french_voice = voice
                        print(f"   ✅ Voix française trouvée: {voice.name}")
                        break
                    elif any(keyword in voice_id for keyword in ['fr-', 'french', 'français']):
                        french_voice = voice
                        print(f"   ✅ Voix française trouvée (ID): {voice.name}")
                        break
                
                if french_voice:
                    self.engine.setProperty('voice', french_voice.id)
                    self.voice_id = french_voice.id
                    print(f"✅ Voix française sélectionnée: {french_voice.name}")
                else:
                    print(f"⚠️ Aucune voix française trouvée, utilisation voix par défaut")
                    self.voice_id = voices[0].id if voices else None
                
                # Configuration
                self.engine.setProperty('rate', 180)  # Vitesse normale
                self.engine.setProperty('volume', 0.9)  # Volume élevé
                
                print(f"✅ SAPI Windows initialisé")
                self.sapi_available = True
                
            except Exception as e:
                print(f"⚠️ Erreur SAPI pyttsx3: {e}")
                self.sapi_available = False
                self._init_win32_sapi()
        else:
            print(f"⚠️ pyttsx3 non disponible")
            self._init_win32_sapi()
            
    def _init_win32_sapi(self):
        """Fallback: utiliser win32com directement"""
        if WIN32_AVAILABLE:
            try:
                self.sapi = win32com.client.Dispatch("SAPI.SpVoice")
                voices = self.sapi.GetVoices()
                
                print(f"🔍 Recherche voix française Win32 parmi {voices.Count} voix...")
                
                for i in range(voices.Count):
                    voice = voices.Item(i)
                    name = voice.GetDescription()
                    print(f"   Voix Win32: {name}")
                    
                    if any(keyword in name.lower() for keyword in ['french', 'français', 'france']):
                        self.sapi.Voice = voice
                        print(f"✅ Voix française Win32 sélectionnée: {name}")
                        break
                
                self.sapi.Rate = 0  # Vitesse normale
                self.sapi.Volume = 90  # Volume élevé
                
                print(f"✅ SAPI Win32 initialisé")
                self.sapi_available = True
                
            except Exception as e:
                print(f"⚠️ Erreur SAPI Win32: {e}")
                self.sapi_available = False
        else:
            print(f"⚠️ win32com non disponible")
            self.sapi_available = False
            
    def synthesize_to_file(self, text, output_path):
        """Synthèse vers fichier WAV"""
        if not self.sapi_available:
            print(f"❌ SAPI non disponible")
            return False
            
        try:
            if hasattr(self, 'engine'):
                # Utiliser pyttsx3
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait()
            elif hasattr(self, 'sapi'):
                # Utiliser win32com
                file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
                file_stream.Open(output_path, 3)  # SSFMCreateForWrite
                self.sapi.AudioOutputStream = file_stream
                self.sapi.Speak(text)
                file_stream.Close()
            else:
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Erreur synthèse fichier: {e}")
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
                
                return audio_data, sample_rate
                
        except Exception as e:
            print(f"❌ Erreur lecture WAV: {e}")
            return None, None
            
    def synthesize(self, text):
        """Synthèse vocale SAPI française"""
        print(f"🇫🇷 Synthèse vocale SAPI FRANÇAISE")
        print(f"   Texte: '{text}'")
        
        if not self.sapi_available:
            print(f"❌ SAPI non disponible")
            return np.array([], dtype=np.int16)
        
        try:
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Synthèse vers fichier
            success = self.synthesize_to_file(text, tmp_path)
            
            if success and os.path.exists(tmp_path):
                # Charger audio
                audio_data, wav_sample_rate = self.load_wav_file(tmp_path)
                
                if audio_data is not None:
                    print(f"   🎵 Audio SAPI français généré: {len(audio_data)} échantillons")
                    print(f"   📊 Sample rate: {wav_sample_rate}Hz")
                    print(f"   🔍 Range audio: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
                    
                    # Vérifier qualité
                    amplitude = max(abs(audio_data.min()), abs(audio_data.max()))
                    if amplitude > 0.01:
                        print(f"   ✅ Audio SAPI français valide (amplitude: {amplitude:.3f})")
                    else:
                        print(f"   ⚠️ Audio SAPI français faible (amplitude: {amplitude:.3f})")
                    
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
            print(f"❌ Erreur synthèse SAPI française: {e}")
            import traceback
            traceback.print_exc()
            return np.array([], dtype=np.int16)
    
    def speak(self, text):
        """Synthèse et lecture audio SAPI français"""
        audio_data = self.synthesize(text)
        
        if len(audio_data) > 0:
            print(f"   🔊 Lecture audio SAPI français...")
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()
            print(f"   ✅ Lecture SAPI française terminée")
        else:
            print(f"   ❌ Pas d'audio SAPI français à lire")
            
        return audio_data 