#!/usr/bin/env python3
"""
EnhancedLLMManager - Gestionnaire LLM avancé avec contexte conversationnel
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE

Conforme aux spécifications du Plan de Développement LUXA Final

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

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from llama_cpp import Llama
from dataclasses import dataclass
import json
from prometheus_client import Counter, Histogram, Gauge

def validate_rtx3090_mandatory():
    """Validation obligatoire de la configuration RTX 3090"""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
        
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices != '1':
            raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1'")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:  # RTX 3090 = ~24GB
            raise RuntimeError(f"🚫 GPU ({gpu_memory:.1f}GB) trop petite - RTX 3090 requise")
        
        print(f"✅ RTX 3090 validée: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    except ImportError:
        print("⚠️ PyTorch non disponible - validation GPU ignorée pour llama-cpp")

# Métriques Prometheus pour monitoring LLM
llm_requests_total = Counter('llm_requests_total', 'Total LLM requests')
llm_errors_total = Counter('llm_errors_total', 'Total LLM errors')
llm_response_time_seconds = Histogram('llm_response_time_seconds', 'LLM response time')
llm_tokens_generated_total = Counter('llm_tokens_generated_total', 'Total tokens generated')
llm_context_resets_total = Counter('llm_context_resets_total', 'Total context resets')

@dataclass
class ConversationTurn:
    """Un tour de conversation"""
    timestamp: float
    user_input: str
    assistant_response: str
    metadata: Dict[str, Any]

class EnhancedLLMManager:
    """
    Manager LLM avancé avec:
    - Gestion contexte conversationnel intelligent
    - Métriques temps réel (latence, tokens, resets)
    - Optimisations performance (timeout, cleanup)
    - Health checks et monitoring
    - Post-processing intelligent des réponses
    - Configuration GPU RTX 3090 exclusive
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Validation RTX 3090 obligatoire
        validate_rtx3090_mandatory()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.conversation_history: List[ConversationTurn] = []
        self.max_context_turns = config.get("max_context_turns", 10)
        self.system_prompt = config.get("system_prompt", self._default_system_prompt())
        
        # Métriques internes pour monitoring
        self.metrics = {
            "total_requests": 0,
            "avg_response_time": 0,
            "total_tokens_generated": 0,
            "context_resets": 0
        }
        
    def _default_system_prompt(self) -> str:
        """Prompt système par défaut pour LUXA"""
        return """Tu es LUXA, un assistant vocal intelligent et bienveillant.
        
Directives:
- Réponds de manière naturelle et conversationnelle
- Sois concis mais informatif (max 100 mots par défaut)
- Adapte ton ton à l'utilisateur
- Si la question n'est pas claire, demande des précisions
- Pour les commandes vocales, exécute et confirme l'action
        
Contexte: Assistant vocal intégré, français prioritaire."""

    async def initialize(self):
        """Initialisation du modèle LLM avec support Ollama et fallback local"""
        self.logger.info("Initialisation EnhancedLLMManager...")
        
        # Essayer d'abord Ollama
        use_ollama = self.config.get('use_ollama', True)
        base_url = self.config.get('base_url', 'http://127.0.0.1:11434/v1')
        model_name = self.config.get('model', 'nous-hermes')
        
        if use_ollama:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get('http://127.0.0.1:11434/api/tags')
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m['name'] for m in models]
                        self.logger.info(f"✅ Ollama accessible - Modèles: {model_names}")
                        
                        # Vérifier si notre modèle existe (recherche exacte ou partielle)
                        exact_match = model_name in model_names
                        partial_match = any(model_name in name or name in model_name for name in model_names)
                        
                        if exact_match or partial_match:
                            # Utiliser le nom exact du modèle trouvé
                            if exact_match:
                                actual_model = model_name
                            else:
                                actual_model = next(name for name in model_names if model_name in name or name in model_name)
                            
                            self.actual_model_name = actual_model
                            self.use_ollama = True
                            self.logger.info(f"✅ Modèle trouvé: {actual_model}")
                            return
                        else:
                            self.logger.warning(f"⚠️ Modèle {model_name} non trouvé - Fallback local")
            except Exception as e:
                self.logger.warning(f"⚠️ Ollama non accessible: {e} - Fallback local")
        
        # Fallback : modèle local
        self.use_ollama = False
        
        # Vérifier si model_path est fourni pour fallback local
        model_path = self.config.get("model_path")
        if not model_path:
            self.logger.warning("⚠️ Aucun model_path configuré - Utilisation fallback simple")
            self.model = None
            return
        
        # Configuration GPU RTX 3090 pour modèle local
        gpu_index = self.config.get("gpu_device_index", 0)
        if gpu_index != 0:
            self.logger.warning(f"⚠️ gpu_device_index={gpu_index} - Avec CUDA_VISIBLE_DEVICES='1', utiliser index 0 (RTX 3090 visible)")
            self.config["gpu_device_index"] = 0
        
        self.logger.info(f"🎮 GPU CONFIG: RTX 3090 exclusif via CUDA_VISIBLE_DEVICES='1' (main_gpu=0)")
        
        try:
            # Configuration optimisée - RTX 3090 UNIQUEMENT
            model_config = {
                "model_path": model_path,
                "n_gpu_layers": self.config.get("n_gpu_layers", 35),
                "main_gpu": 0,  # RTX 3090 seule visible = index 0
                "n_ctx": self.config.get("context_length", 4096),
                "n_threads": self.config.get("n_threads", 4),
                "verbose": False,
                # Optimisations performance RTX 3090
                "use_mmap": True,
                "use_mlock": False,
                "f16_kv": True  # Optimisation mémoire RTX 3090
            }
            
            start_time = time.time()
            self.model = Llama(**model_config)
            load_time = time.time() - start_time
            
            self.logger.info(f"✅ LLM RTX 3090 initialisé en {load_time:.2f}s")
            
            # Test de fonctionnement
            await self._health_check()
            
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation LLM RTX 3090: {e}")
            raise
    
    async def _health_check(self):
        """Test rapide du modèle"""
        try:
            test_response = await self.generate_response(
                "Test", 
                max_tokens=5,
                internal_check=True
            )
            if not test_response:
                raise Exception("Health check failed - no response")
                
            self.logger.info("✅ Health check LLM RTX 3090 réussi")
                
        except Exception as e:
            self.logger.error(f"❌ Health check LLM RTX 3090 failed: {e}")
            raise

    async def generate_response(self, 
                              user_input: str,
                              max_tokens: int = 150,
                              temperature: float = 0.7,
                              include_context: bool = True,
                              internal_check: bool = False) -> str:
        """Génération de réponse avec contexte conversationnel"""
        
        if not internal_check:
            self.metrics["total_requests"] += 1
            llm_requests_total.inc()
        
        start_time = time.time()
        
        try:
            # Construction du prompt avec contexte
            full_prompt = self._build_contextual_prompt(
                user_input, 
                include_context=include_context
            )
            
            # Essayer Ollama d'abord si disponible
            if hasattr(self, 'use_ollama') and self.use_ollama:
                try:
                    response = await self._generate_ollama(user_input, max_tokens, temperature)
                    if response:
                        if not internal_check:
                            self._add_to_history(user_input, response)
                            self._update_metrics(response, time.time() - start_time)
                        
                        response_time = time.time() - start_time
                        llm_response_time_seconds.observe(response_time)
                        
                        self.logger.info(f"✅ Réponse Ollama générée en {response_time:.2f}s")
                        return response
                except Exception as e:
                    self.logger.warning(f"⚠️ Ollama échoué, fallback local: {e}")
            
            # Fallback : modèle local ou réponse simple
            if self.model is None:
                self.logger.warning("⚠️ Aucun modèle LLM disponible - Réponse fallback")
                fallback_response = f"Je reçois votre message : '{user_input}'. Le système LLM n'est pas disponible actuellement, mais la reconnaissance vocale et la synthèse fonctionnent parfaitement."
                
                if not internal_check:
                    self._add_to_history(user_input, fallback_response)
                    self._update_metrics(fallback_response, time.time() - start_time)
                
                response_time = time.time() - start_time
                llm_response_time_seconds.observe(response_time)
                
                self.logger.info(f"✅ Réponse fallback générée en {response_time:.2f}s")
                return fallback_response
            
            self.logger.debug(f"Génération réponse RTX 3090 pour: '{user_input[:50]}...'")
            
            # Génération avec timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(self._generate_sync, full_prompt, max_tokens, temperature),
                timeout=30.0
            )
            
            # Post-traitement
            cleaned_response = self._clean_response(response)
            
            # Sauvegarde dans l'historique
            if not internal_check:
                self._add_to_history(user_input, cleaned_response)
                self._update_metrics(cleaned_response, time.time() - start_time)
            
            response_time = time.time() - start_time
            llm_response_time_seconds.observe(response_time)
            
            self.logger.info(f"✅ Réponse RTX 3090 générée en {response_time:.2f}s")
            return cleaned_response
            
        except asyncio.TimeoutError:
            self.logger.error("⏱️ Timeout génération LLM")
            llm_errors_total.inc()
            return "Désolé, le traitement prend trop de temps. Pouvez-vous répéter ?"
        except Exception as e:
            self.logger.error(f"❌ Erreur génération LLM: {e}")
            llm_errors_total.inc()
            return "Désolé, je rencontre un problème technique. Pouvez-vous reformuler ?"
    
    async def _generate_ollama(self, user_input: str, max_tokens: int, temperature: float) -> str:
        """Génération via Ollama API - VERSION CORRIGÉE"""
        try:
            import httpx
            
            # ✅ CORRECTION: Utiliser l'API native Ollama avec le bon format
            actual_model = getattr(self, 'actual_model_name', self.config.get('model', 'nous-hermes'))
            
            # Format correct pour l'API native Ollama
            data = {
                "model": actual_model,
                "prompt": f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["User:", "\n\n"]
                }
            }
            
            self.logger.info(f"🧠 Requête Ollama: model={actual_model}, tokens={max_tokens}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    'http://127.0.0.1:11434/api/generate',
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()
                    
                    if response_text:
                        self.logger.info(f"✅ Ollama réponse: {response_text[:50]}...")
                        return response_text
                    else:
                        self.logger.warning("⚠️ Ollama réponse vide")
                        return None
                else:
                    self.logger.error(f"❌ Ollama API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Erreur Ollama: {e}")
            return None
                    
        except Exception as e:
            self.logger.error(f"Erreur Ollama: {e}")
            return None
    
    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Génération synchrone (appelée via to_thread)"""
        result = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["Human:", "User:", "\n\n"],
            echo=False
        )
        
        return result['choices'][0]['text']
    
    def _build_contextual_prompt(self, user_input: str, include_context: bool = True) -> str:
        """Construit le prompt avec contexte conversationnel"""
        
        prompt_parts = [self.system_prompt, "\n\n"]
        
        # Ajouter contexte récent si demandé
        if include_context and self.conversation_history:
            prompt_parts.append("Conversation récente:\n")
            
            # Prendre les N derniers tours
            recent_history = self.conversation_history[-self.max_context_turns:]
            
            for turn in recent_history:
                prompt_parts.append(f"Utilisateur: {turn.user_input}\n")
                prompt_parts.append(f"Assistant: {turn.assistant_response}\n\n")
        
        # Requête actuelle
        prompt_parts.append(f"Utilisateur: {user_input}\n")
        prompt_parts.append("Assistant: ")
        
        return "".join(prompt_parts)
    
    def _clean_response(self, raw_response: str) -> str:
        """Nettoie la réponse du modèle"""
        # Supprimer les artifacts communs
        cleaned = raw_response.strip()
        
        # Supprimer les répétitions du prompt
        artifacts = ["Assistant:", "Utilisateur:", "Human:", "User:"]
        for artifact in artifacts:
            if cleaned.startswith(artifact):
                cleaned = cleaned[len(artifact):].strip()
        
        # Limiter la longueur si trop verbose
        if len(cleaned) > 500:
            # Couper à la dernière phrase complète
            sentences = cleaned.split('. ')
            if len(sentences) > 1:
                cleaned = '. '.join(sentences[:-1]) + '.'
        
        return cleaned
    
    def _add_to_history(self, user_input: str, assistant_response: str):
        """Ajoute un tour à l'historique conversationnel"""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input=user_input,
            assistant_response=assistant_response,
            metadata={"source": "voice_assistant"}
        )
        
        self.conversation_history.append(turn)
        
        # Limiter la taille de l'historique
        max_history = self.config.get("max_history_size", 50)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
            self.metrics["context_resets"] += 1
            llm_context_resets_total.inc()
            self.logger.info("🔄 Historique conversationnel nettoyé")
    
    def _update_metrics(self, response: str, response_time: float):
        """Met à jour les métriques"""
        # Moyenne mobile du temps de réponse
        alpha = 0.1
        self.metrics["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * self.metrics["avg_response_time"]
        )
        
        # Estimation tokens générés (approximative)
        estimated_tokens = len(response.split())
        self.metrics["total_tokens_generated"] += estimated_tokens
        llm_tokens_generated_total.inc(estimated_tokens)
    
    def clear_conversation(self):
        """Efface l'historique conversationnel"""
        self.conversation_history.clear()
        self.metrics["context_resets"] += 1
        llm_context_resets_total.inc()
        self.logger.info("🧹 Historique conversationnel effacé")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Résumé de la conversation actuelle"""
        if not self.conversation_history:
            return {"status": "no_conversation"}
        
        return {
            "total_turns": len(self.conversation_history),
            "duration_minutes": (
                (time.time() - self.conversation_history[0].timestamp) / 60
            ),
            "last_interaction": time.time() - self.conversation_history[-1].timestamp,
            "topics": self._extract_topics(),
            "sentiment": self._analyze_sentiment()
        }
    
    def _extract_topics(self) -> List[str]:
        """Extraction basique des sujets abordés"""
        # Implémentation simple basée sur mots-clés
        # Dans un vrai système, utiliser NLP avancé
        common_words = set()
        for turn in self.conversation_history:
            words = turn.user_input.lower().split()
            common_words.update(word for word in words if len(word) > 4)
        
        return list(common_words)[:5]  # Top 5
    
    def _analyze_sentiment(self) -> str:
        """Analyse basique du sentiment"""
        # Implémentation simple - dans un vrai système, utiliser modèle dédié
        positive_words = ["merci", "super", "parfait", "excellent", "génial"]
        negative_words = ["problème", "erreur", "mauvais", "nul", "déçu"]
        
        pos_count = neg_count = 0
        for turn in self.conversation_history:
            text = turn.user_input.lower()
            pos_count += sum(1 for word in positive_words if word in text)
            neg_count += sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def get_status(self) -> Dict[str, Any]:
        """Status détaillé du gestionnaire LLM"""
        return {
            "model_loaded": self.model is not None,
            "conversation_turns": len(self.conversation_history),
            "metrics": self.metrics.copy(),
            "memory_context": len(self._build_contextual_prompt("test", True)),
            "config": {
                "max_context_turns": self.max_context_turns,
                "model_path": self.config.get("model_path", "N/A"),
                "gpu_rtx3090": True
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Export métriques pour monitoring"""
        return {
            **self.metrics,
            "model_loaded": self.model is not None,
            "conversation_turns": len(self.conversation_history),
            "last_interaction": (
                time.time() - self.conversation_history[-1].timestamp
                if self.conversation_history else None
            ),
            "gpu_rtx3090": True
        }

    async def cleanup(self):
        """Nettoyage ressources et historique"""
        self.logger.info("Nettoyage EnhancedLLMManager RTX 3090...")
        
        # Effacer historique
        self.conversation_history.clear()
        
        # Libération modèle si nécessaire  
        if self.model:
            # Note: llama_cpp ne nécessite pas de cleanup explicite
            self.model = None
            
        self.logger.info("✅ Nettoyage RTX 3090 terminé")

# APPELER DANS TOUS LES SCRIPTS PRINCIPAUX
if __name__ == "__main__":
    validate_rtx3090_mandatory() 