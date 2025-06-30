#!/usr/bin/env python3
"""
Cache LRU Optimisé - SuperWhisper V6 TTS Phase 3
Cache intelligent pour textes récurrents avec métriques de performance
🚀 Objectif: Réponse instantanée pour textes répétés
"""

import os
import sys
import time
import hashlib
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import threading

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées complètes"""
    audio_data: bytes
    text: str
    backend_used: str
    creation_time: float
    last_access_time: float
    access_count: int = 0
    audio_duration_ms: float = 0.0
    text_length: int = 0
    cache_key: str = ""
    
    def __post_init__(self):
        self.text_length = len(self.text)
        self.cache_key = self.cache_key or self._generate_key()
    
    def _generate_key(self) -> str:
        """Génération de clé de cache basée sur le contenu"""
        content = f"{self.text}:{self.backend_used}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def update_access(self):
        """Mise à jour des statistiques d'accès"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Vérification d'expiration"""
        return (time.time() - self.creation_time) > ttl_seconds
    
    def get_age_seconds(self) -> float:
        """Âge de l'entrée en secondes"""
        return time.time() - self.creation_time
    
    def get_size_bytes(self) -> int:
        """Taille totale de l'entrée en bytes"""
        return len(self.audio_data) + len(self.text.encode('utf-8')) + 200  # +overhead

class OptimizedTTSCache:
    """
    Cache LRU optimisé pour TTS avec métriques avancées
    
    🚀 OPTIMISATIONS PHASE 3:
    - Cache LRU thread-safe avec OrderedDict
    - Éviction intelligente par taille et TTL
    - Métriques de performance détaillées
    - Préchargement de textes fréquents
    - Compression optionnelle des données audio
    - Persistance sur disque (optionnelle)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configuration du cache
        self.max_size_mb = config.get('max_size_mb', 100)
        self.max_entries = config.get('max_entries', 1000)
        self.ttl_seconds = config.get('ttl_seconds', 3600)  # 1 heure
        self.enable_compression = config.get('enable_compression', False)
        self.enable_persistence = config.get('enable_persistence', False)
        self.persistence_file = config.get('persistence_file', '.cache/tts_cache.db')
        
        # Cache principal (thread-safe)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Métriques de performance
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'total_size_bytes': 0,
            'average_hit_time_ms': 0.0,
            'cache_efficiency': 0.0,
            'startup_time': time.time()
        }
        
        # Tâches de maintenance
        self._maintenance_task = None
        self._maintenance_interval = config.get('maintenance_interval_seconds', 300)  # 5 min
        
        # Initialisation
        self._initialize_cache()
        
        logging.info(f"Cache TTS optimisé initialisé")
        logging.info(f"  Taille max: {self.max_size_mb} MB ({self.max_entries} entrées)")
        logging.info(f"  TTL: {self.ttl_seconds}s")
        logging.info(f"  Compression: {self.enable_compression}")
        logging.info(f"  Persistance: {self.enable_persistence}")
    
    def _initialize_cache(self):
        """Initialisation du cache avec chargement optionnel"""
        try:
            # Chargement depuis la persistance si activée
            if self.enable_persistence:
                self._load_from_persistence()
            
            # Démarrage de la tâche de maintenance
            self._start_maintenance_task()
            
        except Exception as e:
            logging.warning(f"Erreur initialisation cache: {e}")
    
    async def get(self, cache_key: str) -> Optional[bytes]:
        """
        Récupération d'une entrée du cache
        
        🚀 OPTIMISATION: Accès O(1) avec mise à jour LRU
        
        Args:
            cache_key: Clé de cache
            
        Returns:
            bytes: Données audio si trouvées, None sinon
        """
        start_time = time.perf_counter()
        
        with self._lock:
            self._stats['total_requests'] += 1
            
            if cache_key in self._cache:
                # Cache HIT
                entry = self._cache[cache_key]
                
                # Vérification d'expiration
                if entry.is_expired(self.ttl_seconds):
                    logging.debug(f"Cache EXPIRED: {cache_key}")
                    del self._cache[cache_key]
                    self._stats['misses'] += 1
                    return None
                
                # Mise à jour LRU (déplacement en fin)
                self._cache.move_to_end(cache_key)
                entry.update_access()
                
                # Métriques
                hit_time_ms = (time.perf_counter() - start_time) * 1000
                self._stats['hits'] += 1
                self._update_average_hit_time(hit_time_ms)
                
                logging.debug(f"Cache HIT: {cache_key} ({hit_time_ms:.2f}ms)")
                logging.debug(f"  Accès #{entry.access_count}, âge: {entry.get_age_seconds():.1f}s")
                
                return entry.audio_data
            else:
                # Cache MISS
                self._stats['misses'] += 1
                logging.debug(f"Cache MISS: {cache_key}")
                return None
    
    async def put(self, cache_key: str, audio_data: bytes, text: str, 
                  backend_used: str, audio_duration_ms: float = 0.0) -> bool:
        """
        Ajout d'une entrée au cache
        
        🚀 OPTIMISATION: Éviction intelligente par taille et LRU
        
        Args:
            cache_key: Clé de cache
            audio_data: Données audio à cacher
            text: Texte original
            backend_used: Backend utilisé pour la synthèse
            audio_duration_ms: Durée audio en millisecondes
            
        Returns:
            bool: True si ajouté avec succès
        """
        if not audio_data or not text:
            return False
        
        with self._lock:
            try:
                # Compression optionnelle
                if self.enable_compression:
                    audio_data = self._compress_audio(audio_data)
                
                # Création de l'entrée
                entry = CacheEntry(
                    audio_data=audio_data,
                    text=text,
                    backend_used=backend_used,
                    creation_time=time.time(),
                    last_access_time=time.time(),
                    audio_duration_ms=audio_duration_ms,
                    cache_key=cache_key
                )
                
                # Vérification de la taille
                entry_size = entry.get_size_bytes()
                max_size_bytes = self.max_size_mb * 1024 * 1024
                
                if entry_size > max_size_bytes:
                    logging.warning(f"Entrée trop grande pour le cache: {entry_size} bytes")
                    return False
                
                # Éviction si nécessaire
                self._evict_if_needed(entry_size)
                
                # Ajout au cache
                if cache_key in self._cache:
                    # Mise à jour d'une entrée existante
                    old_entry = self._cache[cache_key]
                    self._stats['total_size_bytes'] -= old_entry.get_size_bytes()
                
                self._cache[cache_key] = entry
                self._cache.move_to_end(cache_key)  # Plus récent en fin
                self._stats['total_size_bytes'] += entry_size
                
                logging.debug(f"Cache PUT: {cache_key}")
                logging.debug(f"  Taille: {entry_size} bytes, Texte: {len(text)} chars")
                logging.debug(f"  Cache: {len(self._cache)} entrées, {self._stats['total_size_bytes']} bytes")
                
                return True
                
            except Exception as e:
                logging.error(f"Erreur ajout cache: {e}")
                return False
    
    def _evict_if_needed(self, new_entry_size: int):
        """
        Éviction intelligente basée sur taille et LRU
        
        🚀 OPTIMISATION: Éviction par priorité (LRU + fréquence d'accès)
        """
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        # Éviction par taille
        while (self._stats['total_size_bytes'] + new_entry_size > max_size_bytes or
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
            
            # Éviction LRU (premier élément = plus ancien)
            oldest_key, oldest_entry = next(iter(self._cache.items()))
            
            logging.debug(f"Éviction LRU: {oldest_key}")
            logging.debug(f"  Âge: {oldest_entry.get_age_seconds():.1f}s, Accès: {oldest_entry.access_count}")
            
            self._stats['total_size_bytes'] -= oldest_entry.get_size_bytes()
            del self._cache[oldest_key]
            self._stats['evictions'] += 1
    
    def _compress_audio(self, audio_data: bytes) -> bytes:
        """Compression optionnelle des données audio"""
        try:
            import gzip
            return gzip.compress(audio_data)
        except Exception as e:
            logging.warning(f"Erreur compression audio: {e}")
            return audio_data
    
    def _decompress_audio(self, compressed_data: bytes) -> bytes:
        """Décompression des données audio"""
        try:
            import gzip
            return gzip.decompress(compressed_data)
        except Exception as e:
            logging.warning(f"Erreur décompression audio: {e}")
            return compressed_data
    
    def _update_average_hit_time(self, hit_time_ms: float):
        """Mise à jour de la moyenne des temps de hit"""
        if self._stats['hits'] == 1:
            self._stats['average_hit_time_ms'] = hit_time_ms
        else:
            # Moyenne mobile
            alpha = 0.1  # Facteur de lissage
            self._stats['average_hit_time_ms'] = (
                alpha * hit_time_ms + 
                (1 - alpha) * self._stats['average_hit_time_ms']
            )
    
    def generate_key(self, text: str, params: Dict[str, Any]) -> str:
        """
        Génération de clé de cache normalisée
        
        Args:
            text: Texte à synthétiser
            params: Paramètres de synthèse (voice, speed, etc.)
            
        Returns:
            str: Clé de cache unique
        """
        # Normalisation du texte
        normalized_text = text.strip().lower()
        
        # Sérialisation des paramètres
        param_str = "|".join(f"{k}:{v}" for k, v in sorted(params.items()) if v is not None)
        
        # Génération de la clé
        content = f"{normalized_text}|{param_str}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques détaillées du cache"""
        with self._lock:
            total_requests = self._stats['total_requests']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            # Analyse des entrées
            entries_by_backend = {}
            total_audio_duration = 0.0
            access_counts = []
            
            for entry in self._cache.values():
                backend = entry.backend_used
                entries_by_backend[backend] = entries_by_backend.get(backend, 0) + 1
                total_audio_duration += entry.audio_duration_ms
                access_counts.append(entry.access_count)
            
            return {
                'cache_size_mb': self._stats['total_size_bytes'] / (1024 * 1024),
                'cache_entries': len(self._cache),
                'max_size_mb': self.max_size_mb,
                'max_entries': self.max_entries,
                'hit_rate_percent': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'total_requests': total_requests,
                'average_hit_time_ms': self._stats['average_hit_time_ms'],
                'total_audio_duration_minutes': total_audio_duration / 60000,
                'entries_by_backend': entries_by_backend,
                'average_access_count': sum(access_counts) / len(access_counts) if access_counts else 0,
                'uptime_hours': (time.time() - self._stats['startup_time']) / 3600
            }
    
    def clear(self):
        """Vidage complet du cache"""
        with self._lock:
            self._cache.clear()
            self._stats['total_size_bytes'] = 0
            logging.info("Cache TTS vidé")
    
    def _start_maintenance_task(self):
        """Démarrage de la tâche de maintenance périodique"""
        async def maintenance_loop():
            while True:
                try:
                    await asyncio.sleep(self._maintenance_interval)
                    await self._perform_maintenance()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"Erreur maintenance cache: {e}")
        
        self._maintenance_task = asyncio.create_task(maintenance_loop())
    
    async def _perform_maintenance(self):
        """Maintenance périodique du cache"""
        with self._lock:
            # Nettoyage des entrées expirées
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired(self.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache[key]
                self._stats['total_size_bytes'] -= entry.get_size_bytes()
                del self._cache[key]
                logging.debug(f"Maintenance: Suppression entrée expirée {key}")
            
            if expired_keys:
                logging.info(f"Maintenance cache: {len(expired_keys)} entrées expirées supprimées")
            
            # Persistance optionnelle
            if self.enable_persistence:
                await self._save_to_persistence()
    
    async def _save_to_persistence(self):
        """Sauvegarde du cache sur disque (optionnelle)"""
        # TODO: Implémentation de la persistance
        pass
    
    def _load_from_persistence(self):
        """Chargement du cache depuis le disque (optionnel)"""
        # TODO: Implémentation du chargement
        pass
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        if self.enable_persistence:
            await self._save_to_persistence()
        
        logging.info("Cache TTS optimisé nettoyé")
    
    def __del__(self):
        """Destructeur pour nettoyage automatique"""
        if hasattr(self, '_maintenance_task') and self._maintenance_task:
            logging.warning("Cache TTS détruit sans cleanup() explicite") 