#!/usr/bin/env python3
"""
Test de validation des corrections critiques du double contrôle GPU
Vérifie que les vulnérabilités découvertes ont été corrigées efficacement.

Corrections testées :
1. Fallback sécurisé vers RTX 3090 (GPU 1) même en single-GPU
2. Target GPU inconditionnel (toujours index 1)  
3. Validation VRAM inconditionnelle (24GB requis)
4. Protection absolue contre RTX 5060 (CUDA:0)

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

import unittest
import torch
from unittest.mock import patch, MagicMock
import sys
import os

# Ajouter le chemin du projet pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from STT.stt_manager_robust import RobustSTTManager

class TestDoubleCheckCorrections(unittest.TestCase):
    """Tests de validation des corrections critiques du double contrôle"""
    
    def setUp(self):
        """Configuration initiale pour tous les tests"""
        self.config = {
            'model_cache_dir': './models',
            'fallback_models': ['base', 'small'],
            'use_vad': True,
            'compute_type': 'float16'
        }

    @patch('STT.stt_manager_robust.torch.cuda.is_available')
    @patch('STT.stt_manager_robust.torch.cuda.device_count')
    @patch('STT.stt_manager_robust.torch.cuda.get_device_properties')
    @patch('STT.stt_manager_robust.torch.cuda.memory_allocated')
    @patch('STT.stt_manager_robust.torch.cuda.set_device')
    def test_correction_1_fallback_securise_single_gpu(self, mock_set_device, mock_memory, mock_props, mock_device_count, mock_cuda_available):
        """
        Test Correction 1 : Fallback sécurisé en configuration single-GPU
        AVANT: selected_gpu = 0 (RTX 5060) ❌ [HISTORIQUE - CORRIGÉ]
        APRÈS: selected_gpu = 1 (RTX 3090) ✅
        """
        print("\n🧪 TEST CORRECTION 1 : Fallback sécurisé single-GPU")
        
        # Simuler configuration single-GPU (problématique avant correction)
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1  # Single GPU détectée
        
        # Simuler RTX 3090 (24GB VRAM) sur index 1
        mock_props_instance = MagicMock()
        mock_props_instance.total_memory = 24 * 1024**3  # 24GB
        mock_props.return_value = mock_props_instance
        mock_memory.return_value = 2 * 1024**3  # 2GB utilisés
        
        # Créer instance et vérifier device selection
        manager = RobustSTTManager(self.config)
        device = manager._select_optimal_device()
        
        # VALIDATION CRITIQUE : Doit utiliser CUDA (RTX 3090) même en single-GPU
        self.assertEqual(device, "cuda", "❌ ÉCHEC: Device devrait être 'cuda' avec RTX 3090 détectée")
        
        # VALIDATION : torch.cuda.set_device(1) doit être appelé (RTX 3090)
        mock_set_device.assert_called_with(1)
        print("✅ SUCCÈS: Fallback sécurisé vers GPU 1 (RTX 3090) en single-GPU")
        
        # VALIDATION : Propriétés vérifiées sur GPU 1 (RTX 3090)
        mock_props.assert_called_with(1)
        print("✅ SUCCÈS: Validation VRAM effectuée sur GPU 1 (RTX 3090)")

    @patch('STT.stt_manager_robust.torch.cuda.is_available')
    @patch('STT.stt_manager_robust.torch.cuda.device_count')
    @patch('STT.stt_manager_robust.torch.cuda.get_device_properties')
    @patch('STT.stt_manager_robust.torch.cuda.memory_allocated')
    @patch('STT.stt_manager_robust.torch.cuda.set_device')
    def test_correction_2_target_gpu_inconditionnel(self, mock_set_device, mock_memory, mock_props, mock_device_count, mock_cuda_available):
        """
        Test Correction 2 : Target GPU inconditionnel
        AVANT: target_gpu = 1 if gpu_count >= 2 else 0 ❌ [HISTORIQUE - CORRIGÉ]
        APRÈS: target_gpu = 1 (RTX 3090 toujours) ✅
        """
        print("\n🧪 TEST CORRECTION 2 : Target GPU inconditionnel")
        
        # Test avec dual-GPU
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        mock_props_instance = MagicMock()
        mock_props_instance.total_memory = 24 * 1024**3  # RTX 3090
        mock_props.return_value = mock_props_instance
        mock_memory.return_value = 1 * 1024**3
        
        manager = RobustSTTManager(self.config)
        device = manager._select_optimal_device()
        
        # Dual-GPU : doit utiliser GPU 1 (RTX 3090)
        mock_props.assert_called_with(1)
        print("✅ SUCCÈS: Target GPU = 1 (RTX 3090) en configuration dual-GPU")
        
        # Reset mocks pour test single-GPU
        mock_props.reset_mock()
        mock_device_count.return_value = 1  # Single GPU
        
        manager2 = RobustSTTManager(self.config)
        device2 = manager2._select_optimal_device()
        
        # Single-GPU : doit AUSSI utiliser GPU 1 (RTX 3090)
        mock_props.assert_called_with(1)
        print("✅ SUCCÈS: Target GPU = 1 (RTX 3090) en configuration single-GPU")

    @patch('STT.stt_manager_robust.torch.cuda.is_available')
    @patch('STT.stt_manager_robust.torch.cuda.device_count')
    @patch('STT.stt_manager_robust.torch.cuda.get_device_properties')
    @patch('STT.stt_manager_robust.torch.cuda.memory_allocated')
    def test_correction_3_validation_vram_inconditionnelle(self, mock_memory, mock_props, mock_device_count, mock_cuda_available):
        """
        Test Correction 3 : Validation VRAM inconditionnelle
        AVANT: Validation seulement si gpu_count >= 2 ❌
        APRÈS: Validation systématique toutes configurations ✅
        """
        print("\n🧪 TEST CORRECTION 3 : Validation VRAM inconditionnelle")
        
        mock_cuda_available.return_value = True
        mock_memory.return_value = 1 * 1024**3  # 1GB utilisé
        
        # Test 1 : GPU avec VRAM insuffisante (< 20GB) = RTX 5060 simulée
        print("  📊 Test 3A : GPU insuffisante (RTX 5060 simulée)")
        mock_device_count.return_value = 1
        mock_props_instance = MagicMock()
        mock_props_instance.total_memory = 8 * 1024**3  # 8GB (RTX 5060)
        mock_props.return_value = mock_props_instance
        
        manager = RobustSTTManager(self.config)
        device = manager._select_optimal_device()
        
        # VALIDATION CRITIQUE : Doit fallback vers CPU si VRAM < 20GB
        self.assertEqual(device, "cpu", "❌ ÉCHEC: Devrait fallback CPU avec GPU insuffisante")
        print("✅ SUCCÈS: Fallback CPU avec RTX 5060 simulée (8GB VRAM)")
        
        # Test 2 : GPU avec VRAM suffisante (>= 20GB) = RTX 3090
        print("  📊 Test 3B : GPU suffisante (RTX 3090)")
        mock_props_instance.total_memory = 24 * 1024**3  # 24GB (RTX 3090)
        
        manager2 = RobustSTTManager(self.config)
        device2 = manager2._select_optimal_device()
        
        # VALIDATION : Doit utiliser CUDA avec RTX 3090
        self.assertEqual(device2, "cuda", "❌ ÉCHEC: Devrait utiliser CUDA avec RTX 3090")
        print("✅ SUCCÈS: Utilisation CUDA avec RTX 3090 (24GB VRAM)")

    @patch('STT.stt_manager_robust.torch.cuda.is_available')
    @patch('STT.stt_manager_robust.torch.cuda.device_count')
    @patch('STT.stt_manager_robust.torch.cuda.get_device_properties')
    @patch('STT.stt_manager_robust.torch.cuda.memory_allocated')
    @patch('STT.stt_manager_robust.torch.cuda.set_device')
    def test_correction_4_protection_absolue_rtx5060(self, mock_set_device, mock_memory, mock_props, mock_device_count, mock_cuda_available):
        """
        Test Correction 4 : Protection absolue contre RTX 5060
        Vérifie qu'aucune configuration ne peut utiliser GPU 0 (RTX 5060)
        """
        print("\n🧪 TEST CORRECTION 4 : Protection absolue contre RTX 5060")
        
        mock_cuda_available.return_value = True
        mock_memory.return_value = 1 * 1024**3
        
        # Simuler différentes configurations problématiques
        configurations_test = [
            {"gpu_count": 1, "description": "Single-GPU"},
            {"gpu_count": 2, "description": "Dual-GPU"},
            {"gpu_count": 0, "description": "No-GPU"}
        ]
        
        for config in configurations_test:
            print(f"  📊 Test 4.{config['gpu_count']} : Configuration {config['description']}")
            
            mock_device_count.return_value = config['gpu_count']
            
            if config['gpu_count'] > 0:
                # Simuler RTX 3090 sur index 1
                mock_props_instance = MagicMock()
                mock_props_instance.total_memory = 24 * 1024**3
                mock_props.return_value = mock_props_instance
                
                manager = RobustSTTManager(self.config)
                device = manager._select_optimal_device()
                
                if config['gpu_count'] >= 1:
                    # VALIDATION CRITIQUE : Jamais d'appel à GPU 0
                    set_device_calls = mock_set_device.call_args_list
                    for call in set_device_calls:
                        gpu_index = call[0][0]  # Premier argument de set_device
                        self.assertNotEqual(gpu_index, 0, f"❌ ÉCHEC: GPU 0 (RTX 5060) utilisé en config {config['description']}")
                    
                    # VALIDATION : Propriétés toujours vérifiées sur GPU 1
                    if mock_props.called:
                        props_calls = mock_props.call_args_list
                        for call in props_calls:
                            gpu_index = call[0][0]
                            self.assertEqual(gpu_index, 1, f"❌ ÉCHEC: Propriétés vérifiées sur GPU {gpu_index} au lieu de GPU 1")
                    
                    print(f"✅ SUCCÈS: Aucune utilisation GPU 0 en config {config['description']}")
            
            mock_set_device.reset_mock()
            mock_props.reset_mock()

    def test_correction_integration_complete(self):
        """
        Test d'intégration : Toutes les corrections fonctionnent ensemble
        """
        print("\n🧪 TEST INTÉGRATION : Toutes corrections ensemble")
        
        with patch('STT.stt_manager_robust.torch.cuda.is_available') as mock_cuda, \
             patch('STT.stt_manager_robust.torch.cuda.device_count') as mock_count, \
             patch('STT.stt_manager_robust.torch.cuda.get_device_properties') as mock_props, \
             patch('STT.stt_manager_robust.torch.cuda.memory_allocated') as mock_memory, \
             patch('STT.stt_manager_robust.torch.cuda.set_device') as mock_set:
            
            # Configuration réaliste : RTX 3090 seule disponible
            mock_cuda.return_value = True
            mock_count.return_value = 1  # Single GPU (scénario critique)
            mock_memory.return_value = 2 * 1024**3
            
            mock_props_instance = MagicMock()
            mock_props_instance.total_memory = 24 * 1024**3  # RTX 3090
            mock_props.return_value = mock_props_instance
            
            # Test création instance
            manager = RobustSTTManager(self.config)
            device = manager._select_optimal_device()
            
            # VALIDATIONS INTÉGRÉES
            self.assertEqual(device, "cuda", "❌ Device final incorrect")
            mock_set.assert_called_with(1)  # GPU 1 forcé
            mock_props.assert_called_with(1)  # Propriétés GPU 1
            
            print("✅ SUCCÈS: Intégration complète - RTX 3090 exclusive confirmée")

def run_double_check_validation():
    """Exécute les tests de validation du double contrôle"""
    print("🔍 VALIDATION DES CORRECTIONS DOUBLE CONTRÔLE GPU")
    print("=" * 60)
    print("Validation des 4 corrections critiques appliquées :")
    print("1. Fallback sécurisé single-GPU")
    print("2. Target GPU inconditionnel") 
    print("3. Validation VRAM inconditionnelle")
    print("4. Protection absolue RTX 5060")
    print("=" * 60)
    
    # Créer suite de tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDoubleCheckCorrections)
    
    # Exécuter tests avec output détaillé
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ VALIDATION DOUBLE CONTRÔLE")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ TOUTES LES CORRECTIONS VALIDÉES AVEC SUCCÈS")
        print("🔒 SÉCURITÉ RTX 3090 EXCLUSIVE CONFIRMÉE")
        print("🎯 VULNÉRABILITÉS CRITIQUES ÉLIMINÉES")
    else:
        print("❌ ÉCHECS DÉTECTÉS DANS LES CORRECTIONS")
        print(f"   Erreurs: {len(result.errors)}")
        print(f"   Échecs: {len(result.failures)}")
        
        for test, error in result.errors:
            print(f"   ERREUR {test}: {error}")
        for test, failure in result.failures:
            print(f"   ÉCHEC {test}: {failure}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_double_check_validation()
    exit(0 if success else 1) 