#!/usr/bin/env python3
"""
Utilitaires audio pour SuperWhisper V6 TTS
Conversion PCM → WAV et validation format audio
"""

import io
import wave
import logging

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 22050, channels: int = 1, sampwidth: int = 2) -> bytes:
    """
    Convertit des données PCM brutes (little-endian, signed 16-bit) en WAV avec header RIFF/WAVE.
    
    Args:
        pcm_data: Données PCM brutes
        sample_rate: Fréquence d'échantillonnage (défaut: 22050 Hz)
        channels: Nombre de canaux (défaut: 1 = mono)
        sampwidth: Largeur échantillon en bytes (défaut: 2 = 16-bit)
    
    Returns:
        bytes: Données WAV complètes avec header RIFF/WAVE
    """
    if not pcm_data:
        logging.warning("pcm_to_wav: Données PCM vides")
        return b''
    
    try:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        buf.seek(0)
        wav_data = buf.read()
        
        logging.debug(f"pcm_to_wav: Converti {len(pcm_data)} bytes PCM → {len(wav_data)} bytes WAV")
        return wav_data
        
    except Exception as e:
        logging.error(f"Erreur conversion PCM→WAV: {e}")
        return pcm_data  # Retour données originales en cas d'erreur

def is_valid_wav(data: bytes) -> bool:
    """
    Vérifie la présence des magic bytes 'RIFF' et 'WAVE' pour valider un fichier WAV.
    
    Args:
        data: Données audio à vérifier
    
    Returns:
        bool: True si format WAV valide, False sinon
    """
    if not data or len(data) < 12:
        return False
    
    # Vérification magic bytes RIFF (offset 0) et WAVE (offset 8)
    has_riff = data[:4] == b'RIFF'
    has_wave = data[8:12] == b'WAVE'
    
    is_valid = has_riff and has_wave
    
    if not is_valid:
        logging.debug(f"is_valid_wav: Format invalide - RIFF: {has_riff}, WAVE: {has_wave}")
    
    return is_valid

def get_wav_info(data: bytes) -> dict:
    """
    Extrait les informations d'un fichier WAV (debug/diagnostic).
    
    Args:
        data: Données WAV
    
    Returns:
        dict: Informations du fichier WAV ou erreur
    """
    try:
        if not is_valid_wav(data):
            return {"error": "Format WAV invalide"}
        
        buf = io.BytesIO(data)
        with wave.open(buf, 'rb') as wf:
            info = {
                "channels": wf.getnchannels(),
                "sampwidth": wf.getsampwidth(),
                "framerate": wf.getframerate(),
                "nframes": wf.getnframes(),
                "duration_ms": int(wf.getnframes() / wf.getframerate() * 1000),
                "size_bytes": len(data)
            }
        return info
        
    except Exception as e:
        return {"error": str(e)} 