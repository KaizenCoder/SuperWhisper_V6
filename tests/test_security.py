#!/usr/bin/env python3
"""
Tests de Sécurité LUXA SuperWhisper V6
Suite complète pour validation authentification et sécurité
Phase 1 - Sprint 1 : Tests sécurité de base

🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE - EXÉCUTABLE DEPUIS N'IMPORTE OÙ
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    # Déterminer le répertoire racine du projet
    current_file = pathlib.Path(__file__).resolve()
    
    # Chercher le répertoire racine (contient .git ou marqueurs projet)
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    # Ajouter le projet root au Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Changer le working directory vers project root
    os.chdir(project_root)
    
    # Configuration GPU RTX 3090 obligatoire
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU
    
    print(f"🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")
    print(f"📁 Project Root: {project_root}")
    print(f"💻 Working Directory: {os.getcwd()}")
    
    return project_root

# Initialiser l'environnement portable
_PROJECT_ROOT = _setup_portable_environment()

# Maintenant imports normaux...

import pytest
import asyncio
import time
import os
import tempfile
import hashlib
import hmac
from pathlib import Path
from unittest.mock import patch, MagicMock

# Imports sécurité LUXA
from config.security_config import SecurityConfig, SecurityException, get_security_config
from api.secure_api import app
from fastapi.testclient import TestClient

class TestSecurityConfig:
    """Tests du module de sécurité central"""
    
    @pytest.fixture
    def security_config(self):
        """Fixture configuration sécurité pour tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SecurityConfig(config_path=f"{temp_dir}/test_security.json")
            yield config
    
    def test_api_key_generation_and_validation(self, security_config):
        """Test cycle complet génération → validation clé API"""
        # Génération
        api_key = security_config.generate_api_key("test_user", "Test key")
        
        # Vérifications format
        assert api_key.startswith("luxa_")
        assert len(api_key) > 40  # Longueur minimale sécurisée
        
        # Validation clé valide
        metadata = security_config.validate_api_key(api_key)
        assert metadata is not None
        assert metadata["name"] == "test_user"
        assert metadata["description"] == "Test key"
        assert "created" in metadata
        assert metadata["usage_count"] >= 1
    
    def test_api_key_validation_invalid_keys(self, security_config):
        """Test validation avec clés invalides"""
        # Clé inexistante
        assert security_config.validate_api_key("luxa_invalid_key") is None
        
        # Format invalide
        assert security_config.validate_api_key("invalid_format") is None
        assert security_config.validate_api_key("") is None
        assert security_config.validate_api_key(None) is None
    
    def test_api_key_usage_tracking(self, security_config):
        """Test suivi d'usage des clés API"""
        api_key = security_config.generate_api_key("tracker_test")
        
        # Premier usage
        metadata1 = security_config.validate_api_key(api_key)
        usage_count1 = metadata1["usage_count"]
        last_used1 = metadata1["last_used"]
        
        # Attendre un peu pour timestamp différent
        time.sleep(0.1)
        
        # Deuxième usage
        metadata2 = security_config.validate_api_key(api_key)
        usage_count2 = metadata2["usage_count"]
        last_used2 = metadata2["last_used"]
        
        # Vérifications
        assert usage_count2 > usage_count1
        assert last_used2 != last_used1
    
    def test_api_key_revocation(self, security_config):
        """Test révocation clé API"""
        api_key = security_config.generate_api_key("revoke_test")
        
        # Validation avant révocation
        assert security_config.validate_api_key(api_key) is not None
        
        # Révocation
        revoked = security_config.revoke_api_key(api_key)
        assert revoked is True
        
        # Validation après révocation
        assert security_config.validate_api_key(api_key) is None
        
        # Révocation clé déjà révoquée
        revoked_again = security_config.revoke_api_key(api_key)
        assert revoked_again is False
    
    def test_jwt_token_lifecycle(self, security_config):
        """Test cycle complet JWT : génération → validation → expiration"""
        user_data = {
            "username": "test_user",
            "role": "user",
            "permissions": ["transcription"]
        }
        
        # Génération token
        token = security_config.generate_jwt_token(user_data, expires_hours=1)
        assert isinstance(token, str)
        assert len(token) > 100  # JWT doit être substantiel
        
        # Validation token valide
        payload = security_config.validate_jwt_token(token)
        assert payload["user_data"]["username"] == "test_user"
        assert payload["user_data"]["role"] == "user"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload  # Unique token ID
    
    def test_jwt_token_expiration(self, security_config):
        """Test expiration token JWT"""
        user_data = {"username": "expire_test"}
        
        # Token expiré immédiatement (durée négative)
        with patch('config.security_config.datetime') as mock_datetime:
            from datetime import datetime, timedelta
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1, 12, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            token = security_config.generate_jwt_token(user_data, expires_hours=-1)
            
            # Validation doit échouer
            with pytest.raises(SecurityException) as exc_info:
                security_config.validate_jwt_token(token)
            assert "expiré" in str(exc_info.value)
    
    def test_jwt_token_invalid_signature(self, security_config):
        """Test token JWT avec signature invalide"""
        user_data = {"username": "tamper_test"}
        token = security_config.generate_jwt_token(user_data)
        
        # Modification du token (corruption signature)
        tampered_token = token[:-10] + "tampered12"
        
        with pytest.raises(SecurityException) as exc_info:
            security_config.validate_jwt_token(tampered_token)
        assert "invalide" in str(exc_info.value)
    
    def test_audio_input_validation_valid_formats(self, security_config):
        """Test validation entrées audio avec formats valides"""
        # WAV valide (signature RIFF + données)
        wav_data = b'RIFF\x24\x00\x00\x00WAVEfmt ' + b'\x00' * 100
        result = security_config.validate_audio_input(wav_data, "test.wav")
        assert result["valid"] is True
        assert result["checks"]["size"] is True
        assert result["checks"]["extension"] is True
        assert result["checks"]["magic_bytes"] is True
        assert result["checks"]["malware"] is True
        
        # MP3 valide (signature ID3)
        mp3_data = b'ID3\x03\x00\x00\x00' + b'\x00' * 100
        result = security_config.validate_audio_input(mp3_data, "test.mp3")
        assert result["valid"] is True
        
        # FLAC valide
        flac_data = b'fLaC\x00\x00\x00\x22' + b'\x00' * 100
        result = security_config.validate_audio_input(flac_data, "test.flac")
        assert result["valid"] is True
    
    def test_audio_input_validation_security_checks(self, security_config):
        """Test détection patterns suspects dans audio"""
        # Exécutable Windows (signature MZ)
        malware_data = b'MZ\x90\x00\x03\x00\x00\x00' + b'\x00' * 100
        with pytest.raises(SecurityException) as exc_info:
            security_config.validate_audio_input(malware_data, "malware.wav")
        assert "suspects" in str(exc_info.value)
        
        # Script shell
        script_data = b'#!/bin/bash\necho "hack"' + b'\x00' * 100
        with pytest.raises(SecurityException) as exc_info:
            security_config.validate_audio_input(script_data, "script.wav")
        assert "suspects" in str(exc_info.value)
        
        # JavaScript malveillant
        js_data = b'<script>alert("xss")</script>' + b'\x00' * 100
        with pytest.raises(SecurityException) as exc_info:
            security_config.validate_audio_input(js_data, "evil.wav")
        assert "suspects" in str(exc_info.value)
    
    def test_audio_input_validation_size_limits(self, security_config):
        """Test limites de taille fichiers audio"""
        # Fichier trop volumineux
        large_data = b'\x00' * (26 * 1024 * 1024)  # 26MB > 25MB limite
        with pytest.raises(SecurityException) as exc_info:
            security_config.validate_audio_input(large_data, "large.wav")
        assert "volumineux" in str(exc_info.value)
        
        # Fichier limite acceptable (25MB)
        max_size_data = b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * (25 * 1024 * 1024 - 100)
        result = security_config.validate_audio_input(max_size_data, "max.wav")
        assert result["valid"] is True
    
    def test_audio_input_validation_file_extensions(self, security_config):
        """Test validation extensions fichiers"""
        valid_audio = b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100
        
        # Extensions autorisées
        for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
            result = security_config.validate_audio_input(valid_audio, f"test{ext}")
            assert result["checks"]["extension"] is True
        
        # Extensions interdites
        for ext in [".exe", ".bat", ".sh", ".php", ".js"]:
            with pytest.raises(SecurityException) as exc_info:
                security_config.validate_audio_input(valid_audio, f"malware{ext}")
            assert "non autorisé" in str(exc_info.value)
    
    def test_text_input_sanitization(self, security_config):
        """Test sanitisation entrées texte"""
        # Texte normal
        clean_text = security_config.sanitize_text_input("Bonjour le monde!")
        assert clean_text == "Bonjour le monde!"
        
        # Injection HTML
        html_text = security_config.sanitize_text_input("<script>alert('xss')</script>")
        assert "<script>" not in html_text
        assert "&lt;script&gt;" in html_text
        
        # Caractères de contrôle
        control_text = security_config.sanitize_text_input("Text\x00\x01\x02normal")
        assert "\x00" not in control_text
        assert "Textnormal" in control_text
        
        # Texte trop long
        long_text = "A" * 15000  # > 10k limite
        with pytest.raises(SecurityException) as exc_info:
            security_config.sanitize_text_input(long_text)
        assert "trop long" in str(exc_info.value)
    
    def test_encryption_decryption(self, security_config):
        """Test chiffrement/déchiffrement données sensibles"""
        sensitive_data = "password123!@#"
        
        # Chiffrement
        encrypted = security_config.encrypt_sensitive_data(sensitive_data)
        assert encrypted != sensitive_data
        assert len(encrypted) > len(sensitive_data)
        
        # Déchiffrement
        decrypted = security_config.decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data
        
        # Données corrompues
        with pytest.raises(Exception):  # Fernet lève exception si corrompu
            security_config.decrypt_sensitive_data("corrupted_data")
    
    def test_timing_attack_protection(self, security_config):
        """Test protection contre attaques temporelles"""
        api_key = security_config.generate_api_key("timing_test")
        
        # Mesurer temps validation clé valide vs invalide
        times_valid = []
        times_invalid = []
        
        for _ in range(10):
            start = time.perf_counter()
            security_config.validate_api_key(api_key)
            times_valid.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            security_config.validate_api_key("luxa_invalid_key_same_length_12345")
            times_invalid.append(time.perf_counter() - start)
        
        # Les temps doivent être similaires (protection timing)
        avg_valid = sum(times_valid) / len(times_valid)
        avg_invalid = sum(times_invalid) / len(times_invalid)
        
        # Différence doit être < 50% (protection timing raisonnable)
        time_ratio = abs(avg_valid - avg_invalid) / max(avg_valid, avg_invalid)
        assert time_ratio < 0.5, f"Timing attack possible: {time_ratio:.2%} difference"

class TestSecureAPI:
    """Tests de l'API REST sécurisée"""
    
    @pytest.fixture
    def client(self):
        """Client de test FastAPI"""
        return TestClient(app)
    
    @pytest.fixture
    def valid_jwt_token(self):
        """Token JWT valide pour tests"""
        security = get_security_config()
        user_data = {"username": "test_user", "role": "user"}
        return security.generate_jwt_token(user_data)
    
    @pytest.fixture
    def valid_api_key(self):
        """Clé API valide pour tests"""
        security = get_security_config()
        return security.generate_api_key("test_api_user", "Test API key")
    
    def test_health_endpoints_public(self, client):
        """Test endpoints de santé publics (pas d'auth requise)"""
        # Health check basique
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        
        # Health check sécurité
        response = client.get("/health/security")
        assert response.status_code == 200
        data = response.json()
        assert data["security_status"] == "operational"
    
    def test_auth_token_endpoint(self, client):
        """Test génération token JWT"""
        # Identifiants valides
        response = client.post("/auth/token", json={
            "username": "demo",
            "password": "demo123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 24 * 3600
        
        # Identifiants invalides
        response = client.post("/auth/token", json={
            "username": "wrong",
            "password": "wrong"
        })
        assert response.status_code == 401
    
    def test_api_key_generation_endpoint(self, client, valid_jwt_token):
        """Test génération clé API via endpoint"""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        response = client.post("/auth/api-key", 
                              json={"name": "test_key", "description": "Test"},
                              headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert data["name"] == "test_key"
        assert data["api_key"].startswith("luxa_")
    
    def test_api_key_generation_requires_jwt(self, client):
        """Test que génération API key nécessite JWT (pas API key)"""
        # Sans authentification
        response = client.post("/auth/api-key", json={"name": "test"})
        assert response.status_code == 401
        
        # TODO: Avec API key (doit échouer)
        # api_key = get_security_config().generate_api_key("temp")
        # headers = {"X-API-Key": api_key}
        # response = client.post("/auth/api-key", json={"name": "test"}, headers=headers)
        # assert response.status_code == 401  # API key ne suffit pas
    
    def test_transcription_endpoint_with_jwt(self, client, valid_jwt_token):
        """Test endpoint transcription avec authentification JWT"""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        # Créer fichier audio factice
        audio_content = b'RIFF\x24\x00\x00\x00WAVEfmt ' + b'\x00' * 100
        files = {"audio_file": ("test.wav", audio_content, "audio/wav")}
        
        response = client.post("/api/v1/transcribe", 
                              files=files,
                              headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "confidence" in data
        assert "processing_time" in data
    
    def test_transcription_endpoint_with_api_key(self, client, valid_api_key):
        """Test endpoint transcription avec authentification API Key"""
        headers = {"X-API-Key": valid_api_key}
        
        # Créer fichier audio factice
        audio_content = b'RIFF\x24\x00\x00\x00WAVEfmt ' + b'\x00' * 100
        files = {"audio_file": ("test.wav", audio_content, "audio/wav")}
        
        response = client.post("/api/v1/transcribe", 
                              files=files,
                              headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
    
    def test_transcription_endpoint_without_auth(self, client):
        """Test endpoint transcription sans authentification"""
        audio_content = b'RIFF\x24\x00\x00\x00WAVEfmt ' + b'\x00' * 100
        files = {"audio_file": ("test.wav", audio_content, "audio/wav")}
        
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code == 401
    
    def test_transcription_endpoint_malware_detection(self, client, valid_jwt_token):
        """Test détection malware dans upload audio"""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        # Fichier avec signature exécutable
        malware_content = b'MZ\x90\x00\x03\x00\x00\x00' + b'\x00' * 100
        files = {"audio_file": ("malware.wav", malware_content, "audio/wav")}
        
        response = client.post("/api/v1/transcribe", 
                              files=files,
                              headers=headers)
        assert response.status_code == 422  # Unprocessable Entity
        assert "suspects" in response.json()["detail"]
    
    def test_transcription_endpoint_file_too_large(self, client, valid_jwt_token):
        """Test limite taille fichier"""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        # Fichier > 25MB
        large_content = b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * (26 * 1024 * 1024)
        files = {"audio_file": ("large.wav", large_content, "audio/wav")}
        
        response = client.post("/api/v1/transcribe", 
                              files=files,
                              headers=headers)
        assert response.status_code == 422
        assert "volumineux" in response.json()["detail"]
    
    def test_user_profile_endpoint(self, client, valid_jwt_token, valid_api_key):
        """Test endpoint profil utilisateur"""
        # Avec JWT
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        response = client.get("/api/v1/user/profile", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["authentication_type"] == "jwt"
        
        # Avec API Key
        headers = {"X-API-Key": valid_api_key}
        response = client.get("/api/v1/user/profile", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["authentication_type"] == "api_key"
    
    def test_list_api_keys_jwt_only(self, client, valid_jwt_token, valid_api_key):
        """Test que listing clés API nécessite JWT uniquement"""
        # Avec JWT (doit marcher)
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        response = client.get("/api/v1/user/api-keys", headers=headers)
        assert response.status_code == 200
        
        # Avec API Key (doit échouer pour sécurité)
        headers = {"X-API-Key": valid_api_key}
        response = client.get("/api/v1/user/api-keys", headers=headers)
        assert response.status_code == 401
    
    def test_error_handling_standardized(self, client):
        """Test gestion d'erreurs standardisée"""
        # Endpoint inexistant
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Méthode non autorisée
        response = client.delete("/health")
        assert response.status_code == 405

class TestSecurityPerformance:
    """Tests de performance sécurité"""
    
    def test_validation_performance_under_load(self):
        """Test performance validation sous charge"""
        security = get_security_config()
        api_key = security.generate_api_key("perf_test")
        
        # Test 100 validations rapides
        start_time = time.perf_counter()
        for _ in range(100):
            security.validate_api_key(api_key)
        total_time = time.perf_counter() - start_time
        
        # Validation doit rester <1ms par requête
        avg_time_ms = (total_time / 100) * 1000
        assert avg_time_ms < 1.0, f"Validation trop lente: {avg_time_ms:.2f}ms"
    
    def test_jwt_performance_under_load(self):
        """Test performance JWT sous charge"""
        security = get_security_config()
        user_data = {"username": "perf_test"}
        
        # Génération de tokens
        start_time = time.perf_counter()
        tokens = []
        for i in range(50):
            token = security.generate_jwt_token(user_data)
            tokens.append(token)
        generation_time = time.perf_counter() - start_time
        
        # Validation de tokens
        start_time = time.perf_counter()
        for token in tokens:
            security.validate_jwt_token(token)
        validation_time = time.perf_counter() - start_time
        
        # Performance acceptable
        gen_per_token_ms = (generation_time / 50) * 1000
        val_per_token_ms = (validation_time / 50) * 1000
        
        assert gen_per_token_ms < 10.0, f"Génération JWT trop lente: {gen_per_token_ms:.2f}ms"
        assert val_per_token_ms < 5.0, f"Validation JWT trop lente: {val_per_token_ms:.2f}ms"

# Configuration pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
