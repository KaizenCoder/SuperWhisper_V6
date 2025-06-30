#!/usr/bin/env python3
"""
Handler Piper Daemon - SuperWhisper V6 TTS Phase 3
Pipeline asynchrone avec mode daemon et communication socket
🚀 Performance cible: <50ms (vs 500ms subprocess)
"""

import os
import sys
import asyncio
import logging
import time
import json
import socket
import tempfile
from typing import Optional, Dict, Any, Union
from pathlib import Path
import subprocess

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

from ..utils_audio import pcm_to_wav, is_valid_wav

class PiperDaemonHandler:
    """
    Handler Piper Daemon avec pipeline asynchrone
    
    🚀 OPTIMISATIONS PHASE 3:
    - Démarrage unique du processus Piper au startup
    - Communication via socket UNIX/TCP non-bloquante
    - Pas de fork/exec à chaque synthèse
    - Pipeline asynchrone avec queue de requêtes
    - Reconnexion automatique en cas d'erreur
    
    Performance cible: <50ms (vs 500ms subprocess)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config['model_path']
        self.executable_path = config.get('executable_path', 'piper/piper.exe')
        self.speaker_id = config.get('speaker_id', 0)
        self.target_latency_ms = config.get('target_latency_ms', 50)
        
        # Configuration daemon
        self.daemon_port = config.get('daemon_port', 0)  # 0 = port automatique
        self.daemon_host = config.get('daemon_host', '127.0.0.1')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # État du daemon
        self.daemon_process = None
        self.daemon_socket = None
        self.is_running = False
        self.synthesis_count = 0
        self.startup_time_ms = 0
        
        # Initialisation
        self._validate_configuration()
        
        logging.info(f"Handler Piper Daemon initialisé - Modèle: {Path(self.model_path).name}")
        logging.info(f"Performance cible: <{self.target_latency_ms}ms")
    
    def _validate_configuration(self):
        """Validation de la configuration"""
        if not Path(self.executable_path).exists():
            raise FileNotFoundError(f"Exécutable Piper introuvable: {self.executable_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modèle Piper introuvable: {self.model_path}")
        
        logging.debug(f"✅ Configuration daemon validée")
    
    async def start_daemon(self):
        """
        Démarrage du daemon Piper
        🚀 OPTIMISATION: Processus unique vs fork à chaque appel
        """
        if self.is_running:
            logging.debug("Daemon Piper déjà en cours d'exécution")
            return
        
        start_time = time.perf_counter()
        
        try:
            # Création d'un socket pour la communication
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((self.daemon_host, self.daemon_port))
            server_socket.listen(1)
            
            # Récupération du port assigné automatiquement
            self.daemon_port = server_socket.getsockname()[1]
            
            logging.info(f"🔄 Démarrage daemon Piper...")
            logging.info(f"   Socket: {self.daemon_host}:{self.daemon_port}")
            logging.info(f"   Modèle: {self.model_path}")
            
            # Démarrage du processus Piper en mode serveur
            cmd = [
                self.executable_path,
                "--model", self.model_path,
                "--speaker", str(self.speaker_id),
                "--server-mode",
                "--port", str(self.daemon_port),
                "--output_raw"
            ]
            
            self.daemon_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Attente de la connexion du daemon
            logging.debug("Attente connexion daemon...")
            server_socket.settimeout(10.0)  # Timeout de 10 secondes
            
            # Acceptation de la connexion
            client_socket, addr = server_socket.accept()
            self.daemon_socket = client_socket
            server_socket.close()
            
            self.is_running = True
            self.startup_time_ms = (time.perf_counter() - start_time) * 1000
            
            logging.info(f"✅ Daemon Piper démarré en {self.startup_time_ms:.1f}ms")
            logging.info(f"   PID: {self.daemon_process.pid}")
            logging.info(f"   Connexion: {addr}")
            
        except Exception as e:
            logging.error(f"❌ Échec démarrage daemon Piper: {e}")
            await self.stop_daemon()
            raise RuntimeError(f"Impossible de démarrer le daemon Piper: {e}")
    
    async def stop_daemon(self):
        """Arrêt propre du daemon"""
        if not self.is_running:
            return
        
        logging.info("🔄 Arrêt daemon Piper...")
        
        # Fermeture du socket
        if self.daemon_socket:
            try:
                self.daemon_socket.close()
            except:
                pass
            self.daemon_socket = None
        
        # Arrêt du processus
        if self.daemon_process:
            try:
                self.daemon_process.terminate()
                await asyncio.wait_for(self.daemon_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logging.warning("Timeout arrêt daemon - Force kill")
                self.daemon_process.kill()
                await self.daemon_process.wait()
            except:
                pass
            
            self.daemon_process = None
        
        self.is_running = False
        logging.info("✅ Daemon Piper arrêté")
    
    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """
        Synthèse vocale via daemon asynchrone
        
        🚀 OPTIMISATIONS:
        - Communication socket non-bloquante
        - Pas de fork/exec par appel
        - Pipeline asynchrone
        - Reconnexion automatique
        
        Args:
            text: Texte à synthétiser
            voice: Voix optionnelle (ignorée)
            speed: Vitesse optionnelle (ignorée pour l'instant)
            
        Returns:
            bytes: Audio WAV complet avec headers
        """
        if not text or not text.strip():
            raise ValueError("Texte vide fourni")
        
        # Démarrage automatique du daemon si nécessaire
        if not self.is_running:
            await self.start_daemon()
        
        start_time = time.perf_counter()
        
        for attempt in range(self.max_retries):
            try:
                # 🚀 COMMUNICATION ASYNCHRONE VIA SOCKET
                audio_data = await self._synthesize_via_daemon(text)
                
                # Conversion PCM → WAV si nécessaire
                if not is_valid_wav(audio_data):
                    logging.debug("PiperDaemon: Conversion PCM → WAV")
                    audio_data = pcm_to_wav(
                        pcm_data=audio_data,
                        sample_rate=self.config.get('sample_rate', 22050),
                        channels=self.config.get('channels', 1),
                        sampwidth=2
                    )
                
                # Métriques de performance
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.synthesis_count += 1
                
                # Log de performance
                if latency_ms > self.target_latency_ms:
                    logging.warning(
                        f"Performance Warning: PiperDaemon a dépassé sa cible "
                        f"({latency_ms:.1f}ms > {self.target_latency_ms}ms)"
                    )
                else:
                    logging.debug(
                        f"✅ PiperDaemon: {latency_ms:.1f}ms "
                        f"(cible: {self.target_latency_ms}ms)"
                    )
                
                logging.info(
                    f"PiperDaemon synthèse #{self.synthesis_count}: "
                    f"{len(text)} chars → {len(audio_data)} bytes en {latency_ms:.1f}ms"
                )
                
                return audio_data
                
            except Exception as e:
                logging.warning(f"Tentative {attempt + 1}/{self.max_retries} échouée: {e}")
                
                if attempt < self.max_retries - 1:
                    # Reconnexion pour la prochaine tentative
                    await self.stop_daemon()
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Dernière tentative échouée
                    logging.error(f"Erreur PiperDaemon après {self.max_retries} tentatives: {e}")
                    raise RuntimeError(f"Échec synthèse daemon après {self.max_retries} tentatives: {e}")
    
    async def _synthesize_via_daemon(self, text: str) -> bytes:
        """
        Communication asynchrone avec le daemon via socket
        
        🚀 OPTIMISATION: Socket non-bloquant vs subprocess
        """
        if not self.daemon_socket:
            raise RuntimeError("Socket daemon non disponible")
        
        try:
            # Préparation de la requête JSON
            request = {
                "text": text,
                "speaker_id": self.speaker_id,
                "output_format": "raw"
            }
            request_data = json.dumps(request).encode('utf-8')
            request_length = len(request_data)
            
            # Envoi de la requête (longueur + données)
            header = request_length.to_bytes(4, byteorder='little')
            self.daemon_socket.sendall(header + request_data)
            
            # Réception de la réponse (longueur + données audio)
            response_length_bytes = self._recv_exact(4)
            response_length = int.from_bytes(response_length_bytes, byteorder='little')
            
            if response_length <= 0:
                raise RuntimeError("Réponse daemon invalide")
            
            # Réception des données audio
            audio_data = self._recv_exact(response_length)
            
            return audio_data
            
        except Exception as e:
            raise RuntimeError(f"Erreur communication daemon: {e}")
    
    def _recv_exact(self, length: int) -> bytes:
        """Réception exacte de N bytes via socket"""
        data = b''
        while len(data) < length:
            chunk = self.daemon_socket.recv(length - len(data))
            if not chunk:
                raise RuntimeError("Connexion daemon fermée")
            data += chunk
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du daemon"""
        return {
            'handler_type': 'piper_daemon',
            'is_running': self.is_running,
            'startup_time_ms': self.startup_time_ms,
            'synthesis_count': self.synthesis_count,
            'target_latency_ms': self.target_latency_ms,
            'daemon_port': self.daemon_port,
            'daemon_host': self.daemon_host,
            'process_pid': self.daemon_process.pid if self.daemon_process else None
        }
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        await self.stop_daemon()
        logging.info("PiperDaemonHandler nettoyé")
    
    def __del__(self):
        """Destructeur pour nettoyage automatique"""
        if self.is_running:
            # Note: Ne peut pas utiliser await dans __del__
            logging.warning("PiperDaemonHandler détruit sans cleanup() explicite")


# =============================================================================
# FALLBACK HANDLER POUR COMPATIBILITÉ
# =============================================================================

class PiperDaemonFallbackHandler:
    """
    Handler de fallback si le mode daemon n'est pas supporté
    Utilise l'implémentation CLI standard
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logging.warning(
            "⚠️ Mode daemon Piper non supporté - Utilisation du fallback CLI"
        )
        
        # Import du handler CLI existant
        from ..tts_manager import PiperCliHandler
        self._cli_handler = PiperCliHandler(config)
    
    async def synthesize(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None) -> bytes:
        """Délégation vers le handler CLI"""
        return await self._cli_handler.synthesize(text, voice, speed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du fallback"""
        stats = self._cli_handler.get_stats() if hasattr(self._cli_handler, 'get_stats') else {}
        stats['handler_type'] = 'piper_daemon_fallback'
        return stats
    
    async def cleanup(self):
        """Nettoyage du fallback"""
        if hasattr(self._cli_handler, 'cleanup'):
            await self._cli_handler.cleanup()


# =============================================================================
# FACTORY POUR SÉLECTION AUTOMATIQUE
# =============================================================================

def create_piper_daemon_handler(config: Dict[str, Any]):
    """
    Factory pour créer le handler daemon approprié
    
    Returns:
        PiperDaemonHandler si supporté
        PiperDaemonFallbackHandler sinon
    """
    try:
        return PiperDaemonHandler(config)
    except Exception as e:
        logging.warning(f"Daemon Piper non disponible: {e}")
        return PiperDaemonFallbackHandler(config) 