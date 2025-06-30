#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Sécurité LUXA SuperWhisper V6
Gestionnaire centralisé pour authentification JWT et API Keys
Phase 1 - Sprint 1 : Implémentation sécurité de base
"""

import os
import hashlib
import secrets
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import jwt
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class SecurityException(Exception):
    """Exception sécurisée pour problèmes d'authentification"""
    pass

class SecurityConfig:
    """
    Gestionnaire de sécurité centralisé pour LUXA
    
    Fonctionnalités:
    - Génération/validation clés API sécurisées
    - Gestion tokens JWT avec expiration
    - Validation entrées audio/texte
    - Protection contre attaques timing
    - Chiffrement données sensibles
    """
    
    def __init__(self, config_path: str = "config/security_keys.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        
        # Configuration JWT
        self.jwt_secret = self._get_or_create_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = 24
        
        # Configuration API Keys
        self.api_keys_file = self.config_path.parent / "api_keys.hash"
        self._ensure_api_keys_file()
        
        # Configuration chiffrement
        self.cipher_key = self._get_or_create_cipher_key()
        self.cipher = Fernet(self.cipher_key)
        
        # Limites de sécurité
        self.max_audio_size = 25 * 1024 * 1024  # 25MB
        self.max_text_length = 10000  # 10k caractères
        self.allowed_audio_types = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        logger.info("SecurityConfig initialisé avec succès")

    def _get_or_create_jwt_secret(self) -> str:
        """Génère ou récupère la clé secrète JWT"""
        config_file = self.config_path.parent / "jwt_secret.key"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return f.read().strip()
        else:
            # Génère une nouvelle clé secrète
            secret = secrets.token_urlsafe(64)
            with open(config_file, 'w') as f:
                f.write(secret)
            # Permissions sécurisées (lecture seule propriétaire)
            os.chmod(config_file, 0o600)
            logger.info("Nouvelle clé JWT générée")
            return secret
    
    def _get_or_create_cipher_key(self) -> bytes:
        """Génère ou récupère la clé de chiffrement"""
        key_file = self.config_path.parent / "cipher.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Génère une nouvelle clé de chiffrement
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
            logger.info("Nouvelle clé de chiffrement générée")
            return key
    
    def _ensure_api_keys_file(self):
        """S'assure que le fichier des clés API existe"""
        if not self.api_keys_file.exists():
            with open(self.api_keys_file, 'w') as f:
                f.write("# LUXA API Keys Hash Storage\n")
            os.chmod(self.api_keys_file, 0o600)
    
    def generate_api_key(self, name: str, description: str = "") -> str:
        """
        Génère une nouvelle clé API sécurisée
        
        Args:
            name: Nom identifiant la clé
            description: Description optionnelle
        
        Returns:
            Clé API générée (format: luxa_xxxxx)
        """
        # Génération clé avec préfixe
        api_key = f"luxa_{secrets.token_urlsafe(32)}"
        
        # Hash sécurisé pour stockage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Métadonnées
        metadata = {
            'name': name,
            'description': description,
            'created': datetime.utcnow().isoformat(),
            'last_used': None,
            'usage_count': 0
        }
        
        # Stockage sécurisé
        with open(self.api_keys_file, 'a') as f:
            f.write(f"{key_hash}:{json.dumps(metadata)}\n")
        
        logger.info(f"Clé API générée pour '{name}'")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Valide une clé API avec protection timing attack
        
        Args:
            api_key: Clé API à valider
        
        Returns:
            Métadonnées si valide, None sinon
        """
        if not api_key or not api_key.startswith('luxa_'):
            return None
        
        # Hash de la clé fournie
        provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        try:
            with open(self.api_keys_file, 'r') as f:
                for line in f:
                    if ':' in line and not line.startswith('#'):
                        stored_hash, metadata_json = line.strip().split(':', 1)
                        
                        # Comparaison sécurisée (constant-time)
                        if hmac.compare_digest(provided_hash, stored_hash):
                            metadata = json.loads(metadata_json)
                            
                            # Mise à jour statistiques d'usage
                            self._update_key_usage(stored_hash, metadata)
                            
                            return metadata
        except Exception as e:
            logger.error(f"Erreur validation clé API: {e}")
        
        return None
    
    def _update_key_usage(self, key_hash: str, metadata: Dict):
        """Met à jour les statistiques d'usage d'une clé"""
        metadata['last_used'] = datetime.utcnow().isoformat()
        metadata['usage_count'] = metadata.get('usage_count', 0) + 1
        
        # Rééécriture du fichier (optimisation possible avec base de données)
        lines = []
        try:
            with open(self.api_keys_file, 'r') as f:
                for line in f:
                    if ':' in line and not line.startswith('#'):
                        stored_hash, _ = line.strip().split(':', 1)
                        if stored_hash == key_hash:
                            lines.append(f"{key_hash}:{json.dumps(metadata)}\n")
                        else:
                            lines.append(line)
                    else:
                        lines.append(line)
            
            with open(self.api_keys_file, 'w') as f:
                f.writelines(lines)
                
        except Exception as e:
            logger.error(f"Erreur mise à jour usage clé: {e}")
    
    def generate_jwt_token(self, user_data: Dict, expires_hours: int = None) -> str:
        """
        Génère un token JWT sécurisé
        
        Args:
            user_data: Données utilisateur à inclure
            expires_hours: Durée d'expiration (défaut: 24h)
        
        Returns:
            Token JWT signé
        """
        expires_hours = expires_hours or self.jwt_expiration_hours
        
        payload = {
            'user_data': user_data,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow(),
            'iss': 'luxa-superwhisper-v6',
            'jti': secrets.token_hex(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        logger.info(f"Token JWT généré (expires: {expires_hours}h)")
        
        return token
    
    def validate_jwt_token(self, token: str) -> Dict:
        """
        Valide un token JWT
        
        Args:
            token: Token JWT à valider
        
        Returns:
            Payload décodé si valide
        
        Raises:
            SecurityException: Si token invalide/expiré
        """
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm],
                options={"verify_exp": True, "verify_iat": True}
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise SecurityException("Token JWT expiré")
        except jwt.InvalidTokenError as e:
            raise SecurityException(f"Token JWT invalide: {str(e)}")
    
    def validate_audio_input(self, audio_data: bytes, filename: str = "") -> Dict:
        """
        Validation sécurisée des entrées audio
        
        Args:
            audio_data: Données audio en bytes
            filename: Nom du fichier (optionnel)
        
        Returns:
            Dict avec résultat validation
        
        Raises:
            SecurityException: Si données non sécurisées
        """
        validation_result = {
            'valid': False,
            'size_bytes': len(audio_data),
            'filename': filename,
            'checks': {}
        }
        
        # Vérification taille
        if len(audio_data) > self.max_audio_size:
            raise SecurityException(f"Fichier trop volumineux: {len(audio_data)} > {self.max_audio_size}")
        validation_result['checks']['size'] = True
        
        # Vérification extension
        if filename:
            ext = Path(filename).suffix.lower()
            if ext not in self.allowed_audio_types:
                raise SecurityException(f"Type de fichier non autorisé: {ext}")
            validation_result['checks']['extension'] = True
        
        # Vérification magic bytes (signatures de fichiers)
        magic_bytes_check = self._check_audio_magic_bytes(audio_data)
        validation_result['checks']['magic_bytes'] = magic_bytes_check
        
        # Détection patterns suspects (headers exécutables, scripts)
        malware_check = self._check_malware_patterns(audio_data)
        validation_result['checks']['malware'] = malware_check
        
        if not malware_check:
            raise SecurityException("Patterns suspects détectés dans les données audio")
        
        validation_result['valid'] = True
        logger.info(f"Validation audio réussie: {len(audio_data)} bytes")
        
        return validation_result
    
    def _check_audio_magic_bytes(self, data: bytes) -> bool:
        """Vérifie les signatures de fichiers audio valides"""
        if len(data) < 12:
            return False
        
        # Signatures connues pour formats audio
        audio_signatures = [
            b'RIFF',        # WAV
            b'ID3',         # MP3 avec ID3
            b'\xff\xfb',    # MP3
            b'\xff\xf3',    # MP3
            b'\xff\xf2',    # MP3
            b'fLaC',        # FLAC
            b'OggS',        # OGG
            b'ftypM4A',     # M4A
        ]
        
        for signature in audio_signatures:
            if data.startswith(signature) or signature in data[:12]:
                return True
        
        return False
    
    def _check_malware_patterns(self, data: bytes) -> bool:
        """Détecte les patterns suspects (exécutables, scripts)"""
        # Patterns d'exécutables Windows/Linux
        malware_patterns = [
            b'MZ',          # PE executable
            b'\x7fELF',     # ELF executable
            b'#!/bin/',     # Script shell
            b'<script',     # Script HTML/JS
            b'javascript:', # JavaScript
            b'powershell',  # PowerShell
            b'cmd.exe',     # Windows CMD
        ]
        
        data_lower = data[:1024].lower()  # Check premier KB
        
        for pattern in malware_patterns:
            if pattern.lower() in data_lower:
                logger.warning(f"Pattern suspect détecté: {pattern}")
                return False
        
        return True
    
    def sanitize_text_input(self, text: str) -> str:
        """
        Nettoie et valide les entrées texte
        
        Args:
            text: Texte à nettoyer
        
        Returns:
            Texte nettoyé et sécurisé
        """
        if len(text) > self.max_text_length:
            raise SecurityException(f"Texte trop long: {len(text)} > {self.max_text_length}")
        
        # Suppression caractères de contrôle dangereux
        cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Échappement basique pour prévenir injection
        cleaned = cleaned.replace('<', '&lt;').replace('>', '&gt;')
        
        return cleaned
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Chiffre des données sensibles"""
        encrypted = self.cipher.encrypt(data.encode())
        return encrypted.decode('latin-1')
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Déchiffre des données sensibles"""
        decrypted = self.cipher.decrypt(encrypted_data.encode('latin-1'))
        return decrypted.decode()
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Révoque une clé API
        
        Args:
            api_key: Clé à révoquer
        
        Returns:
            True si révoquée avec succès
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        try:
            lines = []
            revoked = False
            
            with open(self.api_keys_file, 'r') as f:
                for line in f:
                    if ':' in line and not line.startswith('#'):
                        stored_hash, metadata_json = line.strip().split(':', 1)
                        if stored_hash == key_hash:
                            revoked = True
                            continue  # Skip cette ligne (suppression)
                    lines.append(line)
            
            if revoked:
                with open(self.api_keys_file, 'w') as f:
                    f.writelines(lines)
                logger.info("Clé API révoquée avec succès")
            
            return revoked
            
        except Exception as e:
            logger.error(f"Erreur révocation clé API: {e}")
            return False
    
    def list_api_keys(self) -> List[Dict]:
        """Liste toutes les clés API avec métadonnées (sans les clés)"""
        keys_info = []
        
        try:
            with open(self.api_keys_file, 'r') as f:
                for line in f:
                    if ':' in line and not line.startswith('#'):
                        key_hash, metadata_json = line.strip().split(':', 1)
                        metadata = json.loads(metadata_json)
                        metadata['key_hash'] = key_hash[:8] + "..."  # Partial hash pour identification
                        keys_info.append(metadata)
        except Exception as e:
            logger.error(f"Erreur listing clés API: {e}")
        
        return keys_info

# Instance globale singleton
_security_instance = None

def get_security_config() -> SecurityConfig:
    """Récupère l'instance singleton du gestionnaire de sécurité"""
    global _security_instance
    if _security_instance is None:
        _security_instance = SecurityConfig()
    return _security_instance
