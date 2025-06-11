#!/usr/bin/env python3
"""
Gestionnaire d'Erreurs Robuste - Luxa SuperWhisper V6
====================================================

Circuit breaker, retry, et gestion d'erreurs avancée pour tous les composants.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """États du circuit breaker"""
    CLOSED = "closed"           # Normal, pas d'erreurs
    OPEN = "open"              # Trop d'erreurs, circuit ouvert
    HALF_OPEN = "half_open"    # Test si le service est revenu

@dataclass
class ErrorMetrics:
    """Métriques d'erreur pour un componant"""
    total_calls: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    error_types: Dict[str, int] = field(default_factory=dict)
    
    @property
    def error_rate(self) -> float:
        """Taux d'erreur global"""
        if self.total_calls == 0:
            return 0.0
        return self.error_count / self.total_calls
    
    @property
    def recent_error_rate(self) -> float:
        """Taux d'erreur récent (dernières 10 tentatives)"""
        if self.total_calls < 10:
            return self.error_rate
        
        # Approximation simple - dans un vrai système, on garderait l'historique
        recent_errors = min(self.consecutive_errors, 10)
        return recent_errors / 10

class CircuitBreaker:
    """Circuit breaker pour protéger les composants fragiles"""
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 success_threshold: int = 3):
        """
        Args:
            name: Nom du composant protégé
            failure_threshold: Nombre d'échecs avant ouverture
            recovery_timeout: Temps d'attente avant test (secondes)
            success_threshold: Succès nécessaires pour fermer le circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.metrics = ErrorMetrics()
        self.last_state_change = datetime.now()
        self.half_open_success_count = 0
        
        logger.info(f"Circuit breaker '{name}' initialisé: "
                   f"threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Exécute une fonction protégée par le circuit breaker"""
        
        # Vérifier l'état du circuit
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_success_count = 0
                logger.info(f"Circuit breaker '{self.name}' -> HALF_OPEN")
            else:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' ouvert")
        
        # Exécuter la fonction
        start_time = time.time()
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
            
        except Exception as e:
            self._record_failure(e, time.time() - start_time)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Vérifie si on peut tenter de réinitialiser le circuit"""
        if self.state != CircuitState.OPEN:
            return False
        
        time_since_open = datetime.now() - self.last_state_change
        return time_since_open.total_seconds() >= self.recovery_timeout
    
    def _record_success(self, duration: float):
        """Enregistre un succès"""
        self.metrics.total_calls += 1
        self.metrics.consecutive_errors = 0
        self.metrics.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.last_state_change = datetime.now()
                logger.info(f"Circuit breaker '{self.name}' -> CLOSED (recovered)")
    
    def _record_failure(self, error: Exception, duration: float):
        """Enregistre un échec"""
        self.metrics.total_calls += 1
        self.metrics.error_count += 1
        self.metrics.consecutive_errors += 1
        self.metrics.last_error_time = datetime.now()
        
        # Compter par type d'erreur
        error_type = type(error).__name__
        self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
        
        # Vérifier si on doit ouvrir le circuit
        if (self.state == CircuitState.CLOSED and 
            self.metrics.consecutive_errors >= self.failure_threshold):
            
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.warning(f"Circuit breaker '{self.name}' -> OPEN "
                         f"(failures: {self.metrics.consecutive_errors})")
        
        elif self.state == CircuitState.HALF_OPEN:
            # Échec en half-open, retour à open
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.warning(f"Circuit breaker '{self.name}' -> OPEN (half-open failed)")
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état détaillé du circuit breaker"""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "error_count": self.metrics.error_count,
                "error_rate": self.metrics.error_rate,
                "consecutive_errors": self.metrics.consecutive_errors,
                "recent_error_rate": self.metrics.recent_error_rate,
                "error_types": dict(self.metrics.error_types)
            },
            "config": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "success_threshold": self.success_threshold
            },
            "timing": {
                "last_error": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
                "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                "last_state_change": self.last_state_change.isoformat()
            }
        }

class RetryManager:
    """Gestionnaire de retry avec backoff exponentiel"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Exécute une fonction avec retry automatique"""
        
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    # Dernière tentative échouée
                    logger.error(f"Retry failed after {self.max_attempts} attempts: {e}")
                    break
                
                # Calculer le délai de retry
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # Toutes les tentatives ont échoué
        raise last_exception

class CircuitOpenError(Exception):
    """Exception levée quand le circuit breaker est ouvert"""
    pass

class RobustErrorHandler:
    """Gestionnaire d'erreurs centralisé avec circuit breakers et retry"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        self.global_metrics = ErrorMetrics()
        
    def register_component(self, 
                          name: str, 
                          failure_threshold: int = 5,
                          recovery_timeout: int = 60,
                          max_retries: int = 3) -> CircuitBreaker:
        """Enregistre un composant avec son circuit breaker"""
        
        circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        retry_manager = RetryManager(max_attempts=max_retries)
        
        self.circuit_breakers[name] = circuit_breaker
        self.retry_managers[name] = retry_manager
        
        logger.info(f"Composant '{name}' enregistré avec protection")
        return circuit_breaker
    
    async def execute_safe(self, component_name: str, func: Callable, *args, **kwargs) -> Any:
        """Exécute une fonction avec protection complète (circuit breaker + retry)"""
        
        if component_name not in self.circuit_breakers:
            raise ValueError(f"Composant '{component_name}' non enregistré")
        
        circuit_breaker = self.circuit_breakers[component_name]
        retry_manager = self.retry_managers[component_name]
        
        async def protected_call():
            return await circuit_breaker.call(func, *args, **kwargs)
        
        try:
            return await retry_manager.retry(protected_call)
        except Exception as e:
            # Enregistrer l'erreur globale
            self.global_metrics.total_calls += 1
            self.global_metrics.error_count += 1
            
            error_type = type(e).__name__
            self.global_metrics.error_types[error_type] = (
                self.global_metrics.error_types.get(error_type, 0) + 1
            )
            
            logger.error(f"Échec définitif pour '{component_name}': {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Retourne l'état de santé global"""
        
        component_status = {}
        for name, circuit_breaker in self.circuit_breakers.items():
            component_status[name] = circuit_breaker.get_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "global_metrics": {
                "total_calls": self.global_metrics.total_calls,
                "error_count": self.global_metrics.error_count,
                "error_rate": self.global_metrics.error_rate,
                "error_types": dict(self.global_metrics.error_types)
            },
            "components": component_status,
            "healthy_components": sum(1 for cb in self.circuit_breakers.values() 
                                    if cb.state == CircuitState.CLOSED),
            "total_components": len(self.circuit_breakers)
        }
    
    def export_metrics_prometheus(self) -> str:
        """Exporte les métriques au format Prometheus"""
        
        metrics = []
        
        # Métriques globales
        metrics.append(f"luxa_errors_total {self.global_metrics.error_count}")
        metrics.append(f"luxa_calls_total {self.global_metrics.total_calls}")
        metrics.append(f"luxa_error_rate {self.global_metrics.error_rate:.4f}")
        
        # Métriques par composant
        for name, circuit_breaker in self.circuit_breakers.items():
            state_value = {"closed": 0, "open": 1, "half_open": 0.5}[circuit_breaker.state.value]
            
            metrics.append(f'luxa_circuit_breaker_state{{component="{name}"}} {state_value}')
            metrics.append(f'luxa_component_errors_total{{component="{name}"}} {circuit_breaker.metrics.error_count}')
            metrics.append(f'luxa_component_calls_total{{component="{name}"}} {circuit_breaker.metrics.total_calls}')
            metrics.append(f'luxa_component_error_rate{{component="{name}"}} {circuit_breaker.metrics.error_rate:.4f}')
        
        return "\n".join(metrics)

# Instance globale
error_handler = RobustErrorHandler()

# Décorateur pour protection automatique
def protect_component(component_name: str, failure_threshold: int = 5, max_retries: int = 3):
    """Décorateur pour protéger automatiquement une fonction"""
    
    def decorator(func: Callable):
        # Enregistrer le composant si pas déjà fait
        if component_name not in error_handler.circuit_breakers:
            error_handler.register_component(
                component_name, 
                failure_threshold=failure_threshold,
                max_retries=max_retries
            )
        
        async def wrapper(*args, **kwargs):
            return await error_handler.execute_safe(component_name, func, *args, **kwargs)
        
        return wrapper
    return decorator

# Exemple d'utilisation
if __name__ == "__main__":
    import random
    
    # Test du circuit breaker
    async def test_circuit_breaker():
        handler = RobustErrorHandler()
        handler.register_component("test_service", failure_threshold=3, max_retries=2)
        
        async def failing_service():
            if random.random() < 0.7:  # 70% d'échec
                raise Exception("Service indisponible")
            return "Success!"
        
        # Test plusieurs appels
        for i in range(10):
            try:
                result = await handler.execute_safe("test_service", failing_service)
                print(f"Appel {i}: {result}")
            except Exception as e:
                print(f"Appel {i}: Échec - {e}")
            
            await asyncio.sleep(0.5)
        
        # Afficher l'état
        status = handler.get_health_status()
        print(f"\nÉtat final: {json.dumps(status, indent=2)}")
    
    # Lancer le test
    asyncio.run(test_circuit_breaker())
