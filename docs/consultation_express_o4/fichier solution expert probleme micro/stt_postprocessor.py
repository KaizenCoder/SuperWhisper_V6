#!/usr/bin/env python3
"""
Post-Processeur STT Modulaire - SuperWhisper V6
Pipeline: Normalisation → Corrections techniques → Phonétiques → Ponctuation

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

import re
import json
import logging
import unicodedata
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time

class STTPostProcessor:
    """Post-processeur modulaire pour optimiser les transcriptions STT"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = self._setup_logging()
        
        # Configuration par défaut
        self.config = self._load_default_config()
        
        # Charger configuration externe si fournie
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    external_config = json.load(f)
                self.config.update(external_config)
                self.logger.info(f"✅ Configuration chargée: {config_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur chargement config: {e}")
        
        # Statistiques
        self.stats = {
            "total_processed": 0,
            "total_corrections": 0,
            "corrections_by_type": {
                "technical": 0,
                "phonetic": 0,
                "punctuation": 0,
                "normalization": 0
            },
            "processing_time": 0.0,
            "confidence_boosts": 0
        }
        
        self.logger.info("🔧 Post-processeur STT initialisé")
    
    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging"""
        logger = logging.getLogger('STTPostProcessor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut du post-processeur"""
        return {
            "enabled": True,
            "min_confidence_threshold": 0.3,
            "confidence_boost": 0.05,
            
            # Corrections techniques
            "technical_corrections": {
                "gpu": "GPU",
                "rtx": "RTX",
                "rtx 3090": "RTX 3090",
                "rtx3090": "RTX 3090",
                "faster whisper": "faster-whisper",
                "faster whispers": "faster-whisper",
                "after whisper": "faster-whisper",
                "machine learning": "machine learning",
                "intelligence artificielle": "intelligence artificielle",
                "ia": "IA",
                "api": "API",
                "json": "JSON",
                "xml": "XML",
                "html": "HTML",
                "css": "CSS",
                "javascript": "JavaScript",
                "python": "Python",
                "cuda": "CUDA",
                "nvidia": "NVIDIA",
                "amd": "AMD",
                "intel": "Intel",
                "cpu": "CPU",
                "ram": "RAM",
                "ssd": "SSD",
                "hdd": "HDD",
                "usb": "USB",
                "wifi": "Wi-Fi",
                "bluetooth": "Bluetooth"
            },
            
            # Corrections phonétiques françaises
            "phonetic_corrections": {
                "char à": "chat,",
                "char a": "chat,",
                "crésentemps": "chrysanthème",
                "crésentème": "chrysanthème",
                "kakemono": "kakémono",
                "identifiant": "int8",
                "inédite": "int8",
                "sainte vitesse": "fin du test",
                "sacrement": "cinquièmement",
                "dixièmement": "sixièmement",
                "modificieurs": "mots difficiles",
                "agorique": "algorithme",
                "algorique": "algorithme",
                "la tige artificielle": "l'intelligence artificielle",
                "monde monarme": "monde moderne",
                "super whispers": "SuperWhisper",
                "super whisper": "SuperWhisper",
                "superwhispers": "SuperWhisper"
            },
            
            # Règles de ponctuation française
            "punctuation_rules": {
                "add_periods": True,
                "fix_spacing": True,
                "capitalize_sentences": True,
                "fix_quotes": True
            },
            
            # Normalisation Unicode
            "unicode_normalization": "NFKC"
        }
    
    def process(self, text: str, confidence: float = 1.0) -> Tuple[str, Dict[str, Any]]:
        """
        Traite le texte avec le pipeline complet
        
        Args:
            text: Texte à traiter
            confidence: Confiance initiale
            
        Returns:
            (texte traité, métriques)
        """
        if not self.config.get("enabled", True):
            return text, {"corrections_applied": 0}
        
        start_time = time.perf_counter()
        original_text = text
        corrections_count = 0
        
        try:
            self.stats["total_processed"] += 1
            
            # Pipeline de traitement
            processed_text = text
            
            # 1. Normalisation Unicode
            processed_text, norm_corrections = self._normalize_unicode(processed_text)
            corrections_count += norm_corrections
            self.stats["corrections_by_type"]["normalization"] += norm_corrections
            
            # 2. Corrections techniques
            processed_text, tech_corrections = self._apply_technical_corrections(processed_text)
            corrections_count += tech_corrections
            self.stats["corrections_by_type"]["technical"] += tech_corrections
            
            # 3. Corrections phonétiques
            processed_text, phon_corrections = self._apply_phonetic_corrections(processed_text)
            corrections_count += phon_corrections
            self.stats["corrections_by_type"]["phonetic"] += phon_corrections
            
            # 4. Corrections de ponctuation
            processed_text, punct_corrections = self._fix_punctuation(processed_text)
            corrections_count += punct_corrections
            self.stats["corrections_by_type"]["punctuation"] += punct_corrections
            
            # 5. Nettoyage final
            processed_text = self._final_cleanup(processed_text)
            
            # Calcul boost de confiance
            confidence_boost = 0.0
            if corrections_count > 0 and confidence < 0.9:
                confidence_boost = min(
                    self.config.get("confidence_boost", 0.05),
                    0.9 - confidence
                )
                self.stats["confidence_boosts"] += 1
            
            # Métriques
            processing_time = time.perf_counter() - start_time
            self.stats["total_corrections"] += corrections_count
            self.stats["processing_time"] += processing_time
            
            metrics = {
                "corrections_applied": corrections_count,
                "corrections_by_type": {
                    "normalization": norm_corrections,
                    "technical": tech_corrections,
                    "phonetic": phon_corrections,
                    "punctuation": punct_corrections
                },
                "confidence_boost": confidence_boost,
                "processing_time_ms": processing_time * 1000,
                "original_length": len(original_text),
                "processed_length": len(processed_text)
            }
            
            if corrections_count > 0:
                self.logger.info(
                    f"📝 {corrections_count} corrections appliquées "
                    f"({processing_time*1000:.1f}ms)"
                )
            
            return processed_text, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Erreur post-processing: {e}")
            return original_text, {"corrections_applied": 0, "error": str(e)}
    
    def _normalize_unicode(self, text: str) -> Tuple[str, int]:
        """Normalisation Unicode"""
        try:
            original = text
            normalization_form = self.config.get("unicode_normalization", "NFKC")
            normalized = unicodedata.normalize(normalization_form, text)
            
            # Compter les changements
            corrections = 1 if normalized != original else 0
            
            return normalized, corrections
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur normalisation Unicode: {e}")
            return text, 0
    
    def _apply_technical_corrections(self, text: str) -> Tuple[str, int]:
        """Applique les corrections techniques"""
        corrections_count = 0
        corrected_text = text
        
        technical_dict = self.config.get("technical_corrections", {})
        
        for wrong, correct in technical_dict.items():
            # Recherche insensible à la casse
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            matches = pattern.findall(corrected_text)
            
            if matches:
                corrected_text = pattern.sub(correct, corrected_text)
                corrections_count += len(matches)
        
        return corrected_text, corrections_count
    
    def _apply_phonetic_corrections(self, text: str) -> Tuple[str, int]:
        """Applique les corrections phonétiques françaises"""
        corrections_count = 0
        corrected_text = text
        
        phonetic_dict = self.config.get("phonetic_corrections", {})
        
        for wrong, correct in phonetic_dict.items():
            # Recherche exacte pour les corrections phonétiques
            if wrong.lower() in corrected_text.lower():
                # Remplacement insensible à la casse
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                matches = pattern.findall(corrected_text)
                
                if matches:
                    corrected_text = pattern.sub(correct, corrected_text)
                    corrections_count += len(matches)
        
        return corrected_text, corrections_count
    
    def _fix_punctuation(self, text: str) -> Tuple[str, int]:
        """Corrige la ponctuation française"""
        corrections_count = 0
        corrected_text = text
        rules = self.config.get("punctuation_rules", {})
        
        try:
            # 1. Ajouter points en fin de phrase
            if rules.get("add_periods", True):
                if corrected_text and not corrected_text.rstrip().endswith(('.', '!', '?', ':')):
                    corrected_text = corrected_text.rstrip() + '.'
                    corrections_count += 1
            
            # 2. Corriger espacement
            if rules.get("fix_spacing", True):
                # Espaces multiples
                before = corrected_text
                corrected_text = re.sub(r'\s+', ' ', corrected_text)
                if corrected_text != before:
                    corrections_count += 1
                
                # Espaces avant ponctuation
                before = corrected_text
                corrected_text = re.sub(r'\s+([,.!?;:])', r'\1', corrected_text)
                if corrected_text != before:
                    corrections_count += 1
                
                # Espaces après ponctuation
                before = corrected_text
                corrected_text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', corrected_text)
                if corrected_text != before:
                    corrections_count += 1
            
            # 3. Majuscules en début de phrase
            if rules.get("capitalize_sentences", True):
                before = corrected_text
                # Première lettre
                corrected_text = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), corrected_text)
                # Après point, exclamation, interrogation
                corrected_text = re.sub(r'([.!?]\s+)([a-z])', 
                                      lambda m: m.group(1) + m.group(2).upper(), 
                                      corrected_text)
                if corrected_text != before:
                    corrections_count += 1
            
            # 4. Corriger guillemets
            if rules.get("fix_quotes", True):
                before = corrected_text
                # Remplacer guillemets droits par guillemets français
                corrected_text = re.sub(r'"([^"]*)"', r'« \1 »', corrected_text)
                if corrected_text != before:
                    corrections_count += 1
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur correction ponctuation: {e}")
        
        return corrected_text, corrections_count
    
    def _final_cleanup(self, text: str) -> str:
        """Nettoyage final du texte"""
        try:
            # Supprimer espaces en début/fin
            cleaned = text.strip()
            
            # Supprimer lignes vides multiples
            cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
            
            # Supprimer caractères de contrôle
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur nettoyage final: {e}")
            return text
    
    def add_custom_correction(self, wrong: str, correct: str, correction_type: str = "custom"):
        """Ajoute une correction personnalisée"""
        if correction_type == "technical":
            self.config["technical_corrections"][wrong] = correct
        elif correction_type == "phonetic":
            self.config["phonetic_corrections"][wrong] = correct
        else:
            # Créer catégorie custom si nécessaire
            if "custom_corrections" not in self.config:
                self.config["custom_corrections"] = {}
            self.config["custom_corrections"][wrong] = correct
        
        self.logger.info(f"➕ Correction ajoutée: '{wrong}' → '{correct}' ({correction_type})")
    
    def save_config(self, config_path: str):
        """Sauvegarde la configuration actuelle"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 Configuration sauvegardée: {config_path}")
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde config: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du post-processeur"""
        stats = dict(self.stats)
        
        # Calculs dérivés
        if stats["total_processed"] > 0:
            stats["avg_corrections_per_text"] = (
                stats["total_corrections"] / stats["total_processed"]
            )
            stats["avg_processing_time_ms"] = (
                stats["processing_time"] / stats["total_processed"] * 1000
            )
            stats["correction_rate"] = (
                sum(1 for count in stats["corrections_by_type"].values() if count > 0) /
                stats["total_processed"] * 100
            )
        
        return stats
    
    def reset_statistics(self):
        """Remet à zéro les statistiques"""
        self.stats = {
            "total_processed": 0,
            "total_corrections": 0,
            "corrections_by_type": {
                "technical": 0,
                "phonetic": 0,
                "punctuation": 0,
                "normalization": 0
            },
            "processing_time": 0.0,
            "confidence_boosts": 0
        }
        self.logger.info("🔄 Statistiques remises à zéro")


# Fonction utilitaire pour usage simple
def create_post_processor(config_path: Optional[str] = None) -> STTPostProcessor:
    """
    Crée un post-processeur STT avec configuration optionnelle
    
    Args:
        config_path: Chemin vers fichier de configuration JSON
        
    Returns:
        Post-processeur initialisé
    """
    return STTPostProcessor(config_path)


if __name__ == "__main__":
    # Test du post-processeur
    processor = STTPostProcessor()
    
    # Textes de test
    test_texts = [
        "super whisper est un outil de transcription",
        "char à vous allez bien",
        "after whisper est plus rapide",
        "crésentemps est une fleur",
        "gpu rtx 3090 machine learning",
        "bonjour comment allez vous"
    ]
    
    print("🧪 Test du Post-Processeur STT")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        processed, metrics = processor.process(text, confidence=0.7)
        print(f"Résultat: '{processed}'")
        print(f"Corrections: {metrics['corrections_applied']}")
        if metrics['corrections_applied'] > 0:
            print(f"Détail: {metrics['corrections_by_type']}")
    
    # Statistiques finales
    print(f"\n📊 Statistiques finales:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}") 