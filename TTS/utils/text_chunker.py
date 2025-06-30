#!/usr/bin/env python3
"""
Utilitaire de découpage intelligent de texte - SuperWhisper V6 TTS Phase 3
Gestion des textes longs avec chunking sémantique et concaténation WAV
🚀 Objectif: Lever la limite 1000 chars → 5000+ chars

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
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class TextChunk:
    """Représentation d'un chunk de texte avec métadonnées"""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: int
    is_sentence_boundary: bool = False
    estimated_duration_ms: float = 0.0

class IntelligentTextChunker:
    """
    Découpage intelligent de texte pour TTS
    
    🚀 OPTIMISATIONS PHASE 3:
    - Découpage sémantique (phrases, paragraphes)
    - Respect des limites de longueur par backend
    - Préservation de la prosodie naturelle
    - Support des textes 5000+ caractères
    - Estimation de durée par chunk
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Limites par backend
        self.max_chunk_length = config.get('max_chunk_length', 800)  # Sécurité vs 1000
        self.min_chunk_length = config.get('min_chunk_length', 50)   # Éviter chunks trop courts
        self.overlap_chars = config.get('overlap_chars', 20)         # Chevauchement pour fluidité
        
        # Patterns de découpage sémantique
        self.sentence_patterns = [
            r'[.!?]+\s+',           # Fin de phrase standard
            r'[.!?]+$',             # Fin de phrase en fin de texte
            r';\s+',                # Point-virgule
            r':\s+(?=[A-Z])',       # Deux-points suivi de majuscule
        ]
        
        self.paragraph_patterns = [
            r'\n\s*\n',             # Double saut de ligne
            r'\r\n\s*\r\n',         # Double CRLF
        ]
        
        # Estimation de vitesse de parole (chars/seconde)
        self.speech_rate_cps = config.get('speech_rate_cps', 15.0)  # ~900 chars/min
        
        logging.debug(f"TextChunker initialisé - Max: {self.max_chunk_length} chars")
    
    def chunk_text(self, text: str, backend_max_length: Optional[int] = None) -> List[TextChunk]:
        """
        Découpe intelligente d'un texte en chunks optimaux
        
        Args:
            text: Texte à découper
            backend_max_length: Limite spécifique du backend (override config)
            
        Returns:
            List[TextChunk]: Liste des chunks avec métadonnées
        """
        if not text or not text.strip():
            return []
        
        # Utilisation de la limite spécifique ou par défaut
        max_length = backend_max_length or self.max_chunk_length
        
        # Si le texte est déjà assez court, pas de découpage
        if len(text) <= max_length:
            return [TextChunk(
                text=text.strip(),
                start_pos=0,
                end_pos=len(text),
                chunk_id=0,
                is_sentence_boundary=True,
                estimated_duration_ms=self._estimate_duration(text)
            )]
        
        logging.info(f"Découpage texte: {len(text)} chars → chunks de {max_length} chars max")
        
        # Découpage sémantique intelligent
        chunks = self._semantic_chunking(text, max_length)
        
        # Post-traitement et validation
        chunks = self._post_process_chunks(chunks)
        
        logging.info(f"Texte découpé en {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logging.debug(f"  Chunk {i}: {len(chunk.text)} chars, {chunk.estimated_duration_ms:.0f}ms")
        
        return chunks
    
    def _semantic_chunking(self, text: str, max_length: int) -> List[TextChunk]:
        """
        Découpage sémantique en respectant la structure du texte
        
        🚀 OPTIMISATION: Préservation de la prosodie naturelle
        """
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(text):
            # Extraction du chunk optimal
            chunk_text, chunk_end, is_boundary = self._extract_optimal_chunk(
                text, current_pos, max_length
            )
            
            if not chunk_text.strip():
                break
            
            # Création du chunk
            chunk = TextChunk(
                text=chunk_text.strip(),
                start_pos=current_pos,
                end_pos=current_pos + len(chunk_text),
                chunk_id=chunk_id,
                is_sentence_boundary=is_boundary,
                estimated_duration_ms=self._estimate_duration(chunk_text)
            )
            
            chunks.append(chunk)
            
            # Avancement avec chevauchement si nécessaire
            if is_boundary:
                current_pos = chunk_end
            else:
                # Chevauchement pour maintenir la fluidité
                current_pos = max(chunk_end - self.overlap_chars, current_pos + 1)
            
            chunk_id += 1
        
        return chunks
    
    def _extract_optimal_chunk(self, text: str, start_pos: int, max_length: int) -> Tuple[str, int, bool]:
        """
        Extraction d'un chunk optimal en respectant les limites sémantiques
        
        Returns:
            (chunk_text, end_position, is_sentence_boundary)
        """
        remaining_text = text[start_pos:]
        
        if len(remaining_text) <= max_length:
            return remaining_text, len(text), True
        
        # Recherche du meilleur point de coupure
        candidate_text = remaining_text[:max_length]
        
        # 1. Tentative de coupure sur une fin de phrase
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, candidate_text))
            if matches:
                last_match = matches[-1]
                end_pos = start_pos + last_match.end()
                chunk_text = text[start_pos:end_pos]
                return chunk_text, end_pos, True
        
        # 2. Tentative de coupure sur un paragraphe
        for pattern in self.paragraph_patterns:
            matches = list(re.finditer(pattern, candidate_text))
            if matches:
                last_match = matches[-1]
                end_pos = start_pos + last_match.start()
                chunk_text = text[start_pos:end_pos]
                return chunk_text, end_pos, True
        
        # 3. Coupure sur un espace (éviter de couper les mots)
        space_pos = candidate_text.rfind(' ')
        if space_pos > self.min_chunk_length:
            end_pos = start_pos + space_pos
            chunk_text = text[start_pos:end_pos]
            return chunk_text, end_pos, False
        
        # 4. Coupure forcée (dernier recours)
        end_pos = start_pos + max_length
        chunk_text = text[start_pos:end_pos]
        logging.warning(f"Coupure forcée à {max_length} chars (pas de limite sémantique trouvée)")
        return chunk_text, end_pos, False
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Post-traitement des chunks pour optimisation
        
        🚀 OPTIMISATIONS:
        - Fusion des chunks trop courts
        - Nettoyage des espaces
        - Validation des limites
        """
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            # Nettoyage du texte
            cleaned_text = self._clean_chunk_text(chunk.text)
            
            # Skip des chunks vides après nettoyage
            if not cleaned_text.strip():
                continue
            
            # Mise à jour du chunk
            chunk.text = cleaned_text
            chunk.estimated_duration_ms = self._estimate_duration(cleaned_text)
            
            processed_chunks.append(chunk)
        
        # Fusion des chunks trop courts avec le suivant
        final_chunks = self._merge_short_chunks(processed_chunks)
        
        return final_chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Nettoyage du texte d'un chunk"""
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des espaces en début/fin
        text = text.strip()
        
        # Suppression des caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def _merge_short_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Fusion des chunks trop courts pour optimiser la synthèse"""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Si le chunk est trop court et qu'il y a un suivant
            if (len(current_chunk.text) < self.min_chunk_length and 
                i + 1 < len(chunks) and
                len(current_chunk.text) + len(chunks[i + 1].text) <= self.max_chunk_length):
                
                # Fusion avec le chunk suivant
                next_chunk = chunks[i + 1]
                merged_text = current_chunk.text + " " + next_chunk.text
                
                merged_chunk = TextChunk(
                    text=merged_text,
                    start_pos=current_chunk.start_pos,
                    end_pos=next_chunk.end_pos,
                    chunk_id=current_chunk.chunk_id,
                    is_sentence_boundary=next_chunk.is_sentence_boundary,
                    estimated_duration_ms=self._estimate_duration(merged_text)
                )
                
                merged_chunks.append(merged_chunk)
                i += 2  # Skip le chunk suivant
                
                logging.debug(f"Fusion chunks {current_chunk.chunk_id} + {next_chunk.chunk_id}")
            else:
                merged_chunks.append(current_chunk)
                i += 1
        
        return merged_chunks
    
    def _estimate_duration(self, text: str) -> float:
        """
        Estimation de la durée de synthèse en millisecondes
        
        Basé sur la vitesse de parole moyenne
        """
        if not text.strip():
            return 0.0
        
        # Comptage des caractères significatifs (sans espaces multiples)
        clean_text = re.sub(r'\s+', ' ', text.strip())
        char_count = len(clean_text)
        
        # Estimation basée sur la vitesse de parole
        duration_seconds = char_count / self.speech_rate_cps
        
        # Ajout d'une marge pour la synthèse et les pauses
        duration_ms = duration_seconds * 1000 * 1.2  # +20% marge
        
        return duration_ms
    
    def get_total_estimated_duration(self, chunks: List[TextChunk]) -> float:
        """Durée totale estimée pour une liste de chunks"""
        return sum(chunk.estimated_duration_ms for chunk in chunks)
    
    def get_chunking_stats(self, original_text: str, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Statistiques de découpage"""
        if not chunks:
            return {
                'original_length': len(original_text),
                'chunk_count': 0,
                'total_estimated_duration_ms': 0.0,
                'average_chunk_length': 0.0,
                'chunking_efficiency': 0.0
            }
        
        total_chunk_length = sum(len(chunk.text) for chunk in chunks)
        
        return {
            'original_length': len(original_text),
            'chunk_count': len(chunks),
            'total_chunk_length': total_chunk_length,
            'total_estimated_duration_ms': self.get_total_estimated_duration(chunks),
            'average_chunk_length': total_chunk_length / len(chunks),
            'min_chunk_length': min(len(chunk.text) for chunk in chunks),
            'max_chunk_length': max(len(chunk.text) for chunk in chunks),
            'chunking_efficiency': total_chunk_length / len(original_text),
            'sentence_boundaries': sum(1 for chunk in chunks if chunk.is_sentence_boundary)
        }


# =============================================================================
# UTILITAIRES DE CONCATÉNATION AUDIO
# =============================================================================

class AudioConcatenator:
    """
    Concaténation intelligente de fichiers WAV
    
    🚀 OPTIMISATION: Assemblage fluide des chunks audio
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        self.channels = config.get('channels', 1)
        self.silence_duration_ms = config.get('inter_chunk_silence_ms', 200)  # Pause entre chunks
    
    def concatenate_wav_chunks(self, wav_chunks: List[bytes]) -> bytes:
        """
        Concaténation de chunks WAV avec pauses appropriées
        
        Args:
            wav_chunks: Liste des données WAV à concaténer
            
        Returns:
            bytes: Fichier WAV concaténé complet
        """
        if not wav_chunks:
            return b''
        
        if len(wav_chunks) == 1:
            return wav_chunks[0]
        
        logging.debug(f"Concaténation de {len(wav_chunks)} chunks WAV")
        
        try:
            # Import des utilitaires audio
            from ..utils_audio import extract_wav_data, create_wav_header
            
            # Extraction des données audio de chaque chunk
            audio_data_chunks = []
            for i, wav_chunk in enumerate(wav_chunks):
                try:
                    audio_data = extract_wav_data(wav_chunk)
                    audio_data_chunks.append(audio_data)
                except Exception as e:
                    logging.warning(f"Erreur extraction chunk {i}: {e}")
                    continue
            
            if not audio_data_chunks:
                raise RuntimeError("Aucun chunk audio valide")
            
            # Génération du silence inter-chunk
            silence_samples = int(self.silence_duration_ms * self.sample_rate / 1000)
            silence_data = b'\x00\x00' * silence_samples  # Silence 16-bit
            
            # Concaténation avec silences
            concatenated_audio = b''
            for i, audio_data in enumerate(audio_data_chunks):
                concatenated_audio += audio_data
                
                # Ajout de silence entre les chunks (pas après le dernier)
                if i < len(audio_data_chunks) - 1:
                    concatenated_audio += silence_data
            
            # Création du header WAV final
            final_wav = create_wav_header(
                audio_data=concatenated_audio,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sampwidth=2
            ) + concatenated_audio
            
            logging.info(f"Concaténation réussie: {len(wav_chunks)} chunks → {len(final_wav)} bytes")
            return final_wav
            
        except Exception as e:
            logging.error(f"Erreur concaténation WAV: {e}")
            # Fallback: retourner le premier chunk valide
            return wav_chunks[0] if wav_chunks else b'' 