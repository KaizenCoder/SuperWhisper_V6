"""
Tests d'intégration API FastAPI LUXA
- Auth JWT
- Auth API Key
- Endpoint /transcribe (mock)
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

# Import de l'app
from api.secure_api import app, get_authenticated_user, get_current_user_jwt, get_current_user_api_key
from config.security_config import SecurityConfig

# --------------------------------------------------------------------------
# Dépendances mockées
# --------------------------------------------------------------------------

class DummySecurity(SecurityConfig):
    """Sécurité simplifiée pour les tests (pas de fichiers, clés fixes)."""

    def __init__(self):
        super().__init__(config_path="tests/tmp_security.json")
        self.jwt_secret = "testsecret"

    # Surcharge : toujours valider
    def validate_jwt_token(self, token: str):
        return {"user_data": {"username": "test", "role": "tester"}}

    def validate_api_key(self, api_key: str):
        return {"username": "apikey_user"} if api_key == "valid_key" else None


dummy_sec = DummySecurity()

# Override FastAPI dependencies
app.dependency_overrides[get_current_user_jwt] = lambda: {"username": "jwt_user"}
app.dependency_overrides[get_current_user_api_key] = lambda: {"username": "apikey_user"}
app.dependency_overrides[get_authenticated_user] = lambda: {"type": "test", "user": "integration"}

client = TestClient(app)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_transcribe_success_mock():
    """Cas succès : handler mocké renvoie transcription simulée."""
    files = {"audio_file": ("sample.wav", io.BytesIO(b"fake"), "audio/wav")}
    r = client.post("/api/v1/transcribe", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "mock transcription"
    assert body["language"] == "fr"

def test_transcribe_failure_mock():
    """Cas échec simulé : filename contient 'fail'."""
    files = {"audio_file": ("fail_audio.wav", io.BytesIO(b"fake"), "audio/wav")}
    r = client.post("/api/v1/transcribe", files=files)
    assert r.status_code == 500
    assert "Simulated handler failure" in r.json()["detail"]

def test_auth_token_endpoint():
    """Vérifie /auth/token avec les identifiants de démo."""
    data = {"username": "demo", "password": "demo123"}
    r = client.post("/auth/token", data=data)
    assert r.status_code == 200
    payload = r.json()
    assert payload["token_type"] == "bearer"
    assert "access_token" in payload 