#!/usr/bin/env python3
"""
🚨 TEMPLATE BASE POUR TESTS GPU - RTX 3090 OBLIGATOIRE
Base class pour validation GPU homogène SuperWhisper V6
"""

import os
import sys
import unittest
import torch
import functools
import gc
import time
from typing import Optional, Any

# =============================================================================
# 🚨 CONFIGURATION GPU CRITIQUE - RTX 3090 EXCLUSIVEMENT
# =============================================================================
# Configuration physique : RTX 5060 Ti (CUDA:0) + RTX 3090 (CUDA:1)
# RTX 5060 Ti (CUDA:0) = STRICTEMENT INTERDITE
# RTX 3090 (CUDA:1) = SEULE GPU AUTORISÉE VIA MAPPING CUDA:0

# FORCER RTX 3090 EXCLUSIVEMENT
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 Bus PCI 1 → CUDA:0 logique
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre physique stable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

print("🎮 GPU Configuration: RTX 3090 (CUDA:1→CUDA:0) forcée")
print(f"🔒 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"🔒 CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER')}")

def validate_rtx3090_mandatory() -> None:
    """
    Validation OBLIGATOIRE de la configuration RTX 3090
    Lève une exception si RTX 3090 non détectée
    """
    if not torch.cuda.is_available():
        raise RuntimeError("🚫 CUDA non disponible - RTX 3090 requise")
    
    # Vérification variables d'environnement critiques
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices != '1':
        raise RuntimeError(f"🚫 CUDA_VISIBLE_DEVICES='{cuda_devices}' incorrect - doit être '1' (RTX 3090)")
    
    cuda_order = os.environ.get('CUDA_DEVICE_ORDER', '')
    if cuda_order != 'PCI_BUS_ID':
        raise RuntimeError(f"🚫 CUDA_DEVICE_ORDER='{cuda_order}' incorrect - doit être 'PCI_BUS_ID'")
    
    # Vérification GPU physique (RTX 3090 = ~24GB VRAM)
    device_props = torch.cuda.get_device_properties(0)  # CUDA:0 logique = RTX 3090 physique
    gpu_memory_gb = device_props.total_memory / 1024**3
    gpu_name = device_props.name
    
    if gpu_memory_gb < 20:  # RTX 3090 ≈ 24GB, seuil sécurisé 20GB
        raise RuntimeError(f"🚫 GPU ({gpu_name}, {gpu_memory_gb:.1f}GB) insuffisante - RTX 3090 (24GB) requise")
    
    print(f"✅ RTX 3090 validée: {gpu_name} ({gpu_memory_gb:.1f}GB)")
    print(f"✅ Mapping CUDA:0 → RTX 3090 Bus PCI 1 : OPÉRATIONNEL")

def gpu_test_cleanup(func):
    """
    Décorateur pour nettoyage GPU automatique après tests
    Intègre Memory Leak V4.0 - Nettoyage agressif
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Validation obligatoire avant test
            validate_rtx3090_mandatory()
            
            # Exécution du test
            result = func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            print(f"🚫 Erreur test GPU: {e}")
            raise
        finally:
            # Nettoyage agressif Memory Leak V4.0
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                    
                    # Attendre stabilisation GPU
                    time.sleep(0.1)
                    
                    # Vérification mémoire libérée
                    memory_freed = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    if memory_freed > 0:
                        print(f"🧹 GPU cleanup: {memory_freed / 1024**2:.1f}MB libérés")
                        
                except Exception as cleanup_error:
                    print(f"⚠️ Erreur nettoyage GPU: {cleanup_error}")
    
    return wrapper

class GPUCorrectionTestBase(unittest.TestCase):
    """
    Classe de base pour tous les tests de correction GPU
    Assure validation RTX 3090 et nettoyage automatique
    """
    
    @classmethod
    def setUpClass(cls):
        """Setup class unique - Validation RTX 3090 obligatoire"""
        print(f"\n🧪 DÉMARRAGE TESTS GPU: {cls.__name__}")
        print("=" * 50)
        
        # Validation critique RTX 3090
        validate_rtx3090_mandatory()
        
        # Configuration device par défaut
        cls.device = "cuda:0"  # RTX 3090 via mapping CUDA_VISIBLE_DEVICES='1'
        cls.torch_device = torch.device(cls.device)
        
        print(f"🎯 Device configuré: {cls.device} → RTX 3090")
        print(f"🎯 Torch device: {cls.torch_device}")
    
    def setUp(self):
        """Setup avant chaque test - Nettoyage préventif"""
        # Nettoyage préventif
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        # Métriques initiales
        self.initial_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        
    def tearDown(self):
        """Cleanup après chaque test - Memory Leak V4.0"""
        if torch.cuda.is_available():
            try:
                # Nettoyage agressif
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # Vérification memory leak
                final_memory = torch.cuda.memory_allocated(0)
                memory_diff = final_memory - self.initial_memory
                
                if memory_diff > 50 * 1024**2:  # Seuil 50MB
                    print(f"⚠️ Potentiel memory leak détecté: +{memory_diff / 1024**2:.1f}MB")
                else:
                    print(f"✅ Mémoire GPU stable: {memory_diff / 1024**2:.1f}MB")
                    
            except Exception as e:
                print(f"⚠️ Erreur mesure mémoire: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup class final - Nettoyage complet"""
        print(f"\n🏁 FIN TESTS GPU: {cls.__name__}")
        
        if torch.cuda.is_available():
            try:
                # Nettoyage final agressif
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # Statistiques finales
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_reserved = torch.cuda.memory_reserved(0)
                
                print(f"📊 Mémoire finale allouée: {memory_allocated / 1024**2:.1f}MB")
                print(f"📊 Mémoire finale réservée: {memory_reserved / 1024**2:.1f}MB")
                
            except Exception as e:
                print(f"⚠️ Erreur cleanup final: {e}")
        
        print("=" * 50)
    
    def get_gpu_device(self) -> str:
        """Retourne le device GPU configuré (RTX 3090)"""
        return self.device
    
    def get_torch_device(self) -> torch.device:
        """Retourne le torch.device configuré (RTX 3090)"""
        return self.torch_device
    
    def assert_gpu_available(self):
        """Assertion : GPU RTX 3090 disponible"""
        self.assertTrue(torch.cuda.is_available(), "🚫 CUDA/RTX 3090 non disponible")
        
        # Vérifier que c'est bien la RTX 3090
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        self.assertGreaterEqual(gpu_memory, 20, f"🚫 GPU {device_name} ({gpu_memory:.1f}GB) insuffisante - RTX 3090 requise")
        
        print(f"✅ GPU validée dans test: {device_name} ({gpu_memory:.1f}GB)")
    
    @gpu_test_cleanup
    def test_gpu_configuration(self):
        """Test de base : Configuration GPU correcte"""
        # Validation RTX 3090
        self.assert_gpu_available()
        
        # Vérification variables d'environnement
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '1')
        self.assertEqual(os.environ.get('CUDA_DEVICE_ORDER'), 'PCI_BUS_ID')
        
        # Test device fonctionnel
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(self.torch_device)
        self.assertEqual(test_tensor.device.type, 'cuda')
        self.assertEqual(test_tensor.device.index, 0)  # CUDA:0 logique = RTX 3090
        
        print("✅ Test configuration GPU: SUCCÈS")

# Fonction utilitaire pour validation rapide
def quick_gpu_validation():
    """Validation rapide RTX 3090 - Utilisable dans n'importe quel script"""
    try:
        validate_rtx3090_mandatory()
        print("✅ Validation GPU rapide: RTX 3090 opérationnelle")
        return True
    except Exception as e:
        print(f"🚫 Validation GPU rapide: ÉCHEC - {e}")
        return False

if __name__ == "__main__":
    # Test autonome de la configuration
    print("🧪 TEST AUTONOME - Template Base GPU")
    print("=" * 50)
    
    # Validation RTX 3090
    validate_rtx3090_mandatory()
    
    # Tests de base
    suite = unittest.TestLoader().loadTestsFromTestCase(GPUCorrectionTestBase)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🎯 TEMPLATE GPU BASE: VALIDATION RÉUSSIE")
        sys.exit(0)
    else:
        print("\n🚫 TEMPLATE GPU BASE: ÉCHEC VALIDATION")
        sys.exit(1) 