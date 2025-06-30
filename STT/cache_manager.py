#!/usr/bin/env python3
"""
Cache Manager STT - SuperWhisper V6 Phase 4
Cache LRU pour résultats de transcription avec TTL et métriques

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

import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées"""
    value: Dict[str, Any]
    timestamp: float
    size: int
    access_count: int = 0
    last_access: float = 0.0

class STTCache:
    """Cache LRU pour résultats STT avec TTL et surveillance"""
    
    def __init__(self, max_size: int = 200*1024*1024, ttl: int = 7200):
        """
        Initialise le cache LRU.
        
        Args:
            max_size: Taille maximale en bytes (défaut: 200MB)
            ttl: Durée de vie des entrées en secondes (défaut: 2h)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # {key: CacheEntry}
        self.current_size = 0
        
        # Métriques
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_cleanups = 0
        
        print(f"🗄️ Cache STT initialisé: {max_size/1024/1024:.0f}MB, TTL={ttl}s")
    
    def _generate_cache_key(self, audio: np.ndarray, config: Dict[str, Any] = None) -> str:
        """
        Génère une clé de cache unique pour l'audio et la configuration.
        
        Args:
            audio: Données audio numpy
            config: Configuration de transcription (optionnel)
            
        Returns:
            Clé de cache MD5
        """
        # Hash de l'audio
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        
        # Hash de la configuration si fournie
        if config:
            config_str = str(sorted(config.items()))
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            return f"{audio_hash}_{config_hash}"
        
        return audio_hash
    
    def _estimate_size(self, value: Dict[str, Any]) -> int:
        """
        Estime la taille d'une entrée de cache.
        
        Args:
            value: Valeur à stocker
            
        Returns:
            Taille estimée en bytes
        """
        # Estimation basique - peut être affinée
        text_size = len(value.get('text', '')) * 2  # UTF-8
        segments_size = len(str(value.get('segments', []))) * 2
        metadata_size = 200  # Estimation pour les autres champs
        
        return text_size + segments_size + metadata_size
    
    def _cleanup_expired(self):
        """Nettoie les entrées expirées"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self.cache.pop(key)
            self.current_size -= entry.size
            self.expired_cleanups += 1
        
        if expired_keys:
            print(f"🧹 Cache: {len(expired_keys)} entrées expirées nettoyées")
    
    def _evict_lru(self, needed_space: int):
        """
        Évince les entrées LRU pour libérer l'espace nécessaire.
        
        Args:
            needed_space: Espace requis en bytes
        """
        freed_space = 0
        
        while freed_space < needed_space and self.cache:
            # Retire l'entrée la moins récemment utilisée
            key, entry = self.cache.popitem(last=False)
            freed_space += entry.size
            self.current_size -= entry.size
            self.evictions += 1
        
        if freed_space > 0:
            print(f"🗑️ Cache: {freed_space/1024:.0f}KB libérés par éviction LRU")
    
    def get(self, audio: np.ndarray, config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Récupère un résultat du cache.
        
        Args:
            audio: Données audio pour générer la clé
            config: Configuration de transcription
            
        Returns:
            Résultat mis en cache ou None si absent/expiré
        """
        key = self._generate_cache_key(audio, config)
        current_time = time.time()
        
        # Nettoyage périodique des entrées expirées
        if len(self.cache) % 100 == 0:  # Tous les 100 accès
            self._cleanup_expired()
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Vérification TTL
            if current_time - entry.timestamp <= self.ttl:
                # Déplace vers la fin (plus récent)
                self.cache.move_to_end(key)
                
                # Met à jour les statistiques d'accès
                entry.access_count += 1
                entry.last_access = current_time
                
                self.hits += 1
                return entry.value
            else:
                # Entrée expirée
                self.cache.pop(key)
                self.current_size -= entry.size
                self.expired_cleanups += 1
        
        self.misses += 1
        return None
    
    def put(self, audio: np.ndarray, value: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Stocke un résultat dans le cache.
        
        Args:
            audio: Données audio pour générer la clé
            value: Résultat de transcription à cacher
            config: Configuration de transcription
        """
        key = self._generate_cache_key(audio, config)
        entry_size = self._estimate_size(value)
        current_time = time.time()
        
        # Vérification de l'espace disponible
        if entry_size > self.max_size:
            print(f"⚠️ Cache: Entrée trop grande ({entry_size/1024:.0f}KB > {self.max_size/1024/1024:.0f}MB)")
            return
        
        # Libération d'espace si nécessaire
        needed_space = entry_size - (self.max_size - self.current_size)
        if needed_space > 0:
            self._evict_lru(needed_space)
        
        # Suppression de l'ancienne entrée si elle existe
        if key in self.cache:
            old_entry = self.cache.pop(key)
            self.current_size -= old_entry.size
        
        # Ajout de la nouvelle entrée
        entry = CacheEntry(
            value=value,
            timestamp=current_time,
            size=entry_size,
            access_count=1,
            last_access=current_time
        )
        
        self.cache[key] = entry
        self.current_size += entry_size
        
        print(f"💾 Cache: Nouvelle entrée {entry_size/1024:.0f}KB (total: {self.current_size/1024/1024:.1f}MB)")
    
    def clear(self):
        """Vide complètement le cache"""
        self.cache.clear()
        self.current_size = 0
        print("🗑️ Cache STT vidé")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dict avec métriques du cache
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "entries": len(self.cache),
            "current_size_mb": self.current_size / 1024 / 1024,
            "max_size_mb": self.max_size / 1024 / 1024,
            "usage_percent": (self.current_size / self.max_size) * 100,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "expired_cleanups": self.expired_cleanups,
            "ttl_seconds": self.ttl
        }
    
    def get_top_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retourne les entrées les plus utilisées.
        
        Args:
            limit: Nombre d'entrées à retourner
            
        Returns:
            Liste des entrées triées par nombre d'accès
        """
        entries = []
        
        for key, entry in self.cache.items():
            entries.append({
                "key": key[:16] + "...",  # Clé tronquée
                "access_count": entry.access_count,
                "size_kb": entry.size / 1024,
                "age_minutes": (time.time() - entry.timestamp) / 60,
                "last_access_minutes": (time.time() - entry.last_access) / 60
            })
        
        # Tri par nombre d'accès décroissant
        entries.sort(key=lambda x: x["access_count"], reverse=True)
        
        return entries[:limit]
    
    def optimize(self):
        """Optimise le cache en nettoyant les entrées expirées"""
        initial_size = len(self.cache)
        self._cleanup_expired()
        cleaned = initial_size - len(self.cache)
        
        if cleaned > 0:
            print(f"🔧 Cache optimisé: {cleaned} entrées expirées supprimées")
        
        return cleaned

# Fonction utilitaire pour créer un cache configuré
def create_stt_cache(max_size_mb: int = 200, ttl_hours: int = 2) -> STTCache:
    """
    Crée un cache STT configuré.
    
    Args:
        max_size_mb: Taille maximale en MB
        ttl_hours: Durée de vie en heures
        
    Returns:
        Instance STTCache configurée
    """
    return STTCache(
        max_size=max_size_mb * 1024 * 1024,
        ttl=ttl_hours * 3600
    )

# Point d'entrée pour tests
if __name__ == "__main__":
    print("🧪 Test STTCache")
    
    # Création du cache
    cache = create_stt_cache(max_size_mb=10, ttl_hours=1)
    
    # Test avec audio factice
    test_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)  # 1s
    test_result = {
        "text": "Ceci est un test de transcription",
        "confidence": 0.95,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Ceci est un test"}],
        "processing_time": 0.1
    }
    
    # Test de stockage
    print("💾 Test stockage...")
    cache.put(test_audio, test_result)
    
    # Test de récupération
    print("🔍 Test récupération...")
    cached_result = cache.get(test_audio)
    
    if cached_result:
        print(f"✅ Cache hit: '{cached_result['text']}'")
    else:
        print("❌ Cache miss")
    
    # Affichage des statistiques
    stats = cache.get_stats()
    print(f"📊 Statistiques: {stats}")
    
    # Test avec audio différent
    test_audio2 = np.random.normal(0, 0.2, 32000).astype(np.float32)  # 2s
    cached_result2 = cache.get(test_audio2)
    
    if cached_result2:
        print("❌ Erreur: Cache hit inattendu")
    else:
        print("✅ Cache miss attendu pour audio différent")
    
    print("�� Tests terminés") 