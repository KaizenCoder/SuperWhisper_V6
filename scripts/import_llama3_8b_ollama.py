#!/usr/bin/env python3
"""
📥 IMPORT LLAMA3 8B DANS OLLAMA - SUPERWHISPER V6
===============================================
Script pour importer le modèle Llama3 8B Instruct Q6_K dans Ollama

MODÈLE SOURCE:
- Fichier: Meta-Llama-3-8B-Instruct-Q6_K.gguf
- Localisation: D:\modeles_llm\lmstudio-community\Meta-Llama-3-8B-Instruct-GGUF\
- Taille: ~6.1GB

ACTIONS:
1. Création Modelfile Ollama
2. Import modèle depuis fichier local
3. Validation modèle chargé
4. Test rapide inférence

Usage: python scripts/import_llama3_8b_ollama.py

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

import subprocess
import time
import logging
import tempfile
import httpx
import asyncio

# Configuration logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("llama3_import")

class Llama3OllamaImporter:
    """Importateur Llama3 8B dans Ollama"""
    
    def __init__(self):
        self.model_name = "llama3:8b-instruct-q6_k"
        self.model_path = r"D:\modeles_llm\lmstudio-community\Meta-Llama-3-8B-Instruct-GGUF\Meta-Llama-3-8B-Instruct-Q6_K.gguf"
        self.ollama_base_url = "http://localhost:11434"
        
        # Template Modelfile pour Llama3 8B Instruct
        self.modelfile_template = """FROM {model_path}

# Paramètres optimisés pour conversation française
PARAMETER temperature 0.3
PARAMETER top_p 0.8
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 50

# Template Llama3 Instruct
TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un assistant vocal français. Réponds de manière très concise et naturelle.<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"

# Messages système
SYSTEM \"Tu es un assistant vocal français. Réponds de manière très concise et naturelle.\"

# Paramètres d'arrêt
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""
    
    def check_model_file_exists(self) -> bool:
        """Vérification existence fichier modèle"""
        if os.path.exists(self.model_path):
            file_size_gb = os.path.getsize(self.model_path) / (1024**3)
            logger.info(f"✅ Fichier modèle trouvé: {file_size_gb:.1f}GB")
            return True
        else:
            logger.error(f"❌ Fichier modèle non trouvé: {self.model_path}")
            return False
    
    async def check_ollama_available(self) -> bool:
        """Vérification Ollama disponible"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    logger.info("✅ Ollama disponible")
                    return True
                else:
                    logger.error(f"❌ Ollama erreur: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ Ollama non disponible: {e}")
            return False
    
    async def check_model_already_loaded(self) -> bool:
        """Vérification si modèle déjà chargé"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    
                    if self.model_name in models:
                        logger.info(f"✅ Modèle {self.model_name} déjà chargé")
                        return True
                    else:
                        logger.info(f"📋 Modèles disponibles: {models}")
                        logger.info(f"❌ Modèle {self.model_name} non trouvé")
                        return False
        except Exception as e:
            logger.error(f"❌ Erreur vérification modèles: {e}")
            return False
    
    def create_modelfile(self) -> str:
        """Création Modelfile temporaire"""
        modelfile_content = self.modelfile_template.format(model_path=self.model_path)
        
        # Création fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name
        
        logger.info(f"📝 Modelfile créé: {modelfile_path}")
        return modelfile_path
    
    def import_model_ollama(self, modelfile_path: str) -> bool:
        """Import modèle dans Ollama via CLI"""
        try:
            logger.info(f"📥 Import modèle {self.model_name} dans Ollama...")
            logger.info("⏳ Cette opération peut prendre plusieurs minutes...")
            
            # Commande ollama create
            cmd = ["ollama", "create", self.model_name, "-f", modelfile_path]
            
            logger.info(f"🔧 Commande: {' '.join(cmd)}")
            
            # Exécution avec capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )
            
            if result.returncode == 0:
                logger.info("✅ Import réussi!")
                logger.info(f"📤 Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Import échoué (code {result.returncode})")
                logger.error(f"📤 Stdout: {result.stdout}")
                logger.error(f"📤 Stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Import timeout (>10min)")
            return False
        except FileNotFoundError:
            logger.error("❌ Commande 'ollama' non trouvée - Ollama CLI requis")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur import: {e}")
            return False
    
    async def test_imported_model(self) -> bool:
        """Test rapide modèle importé"""
        try:
            logger.info("🧪 Test modèle importé...")
            
            payload = {
                "model": self.model_name,
                "prompt": "Bonjour !",
                "stream": False,
                "options": {
                    "num_predict": 10
                }
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '').strip()
                    logger.info(f"✅ Test réussi: '{response_text}'")
                    return True
                else:
                    logger.error(f"❌ Test échoué: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur test: {e}")
            return False
    
    def cleanup_modelfile(self, modelfile_path: str):
        """Nettoyage fichier temporaire"""
        try:
            os.unlink(modelfile_path)
            logger.info("🧹 Modelfile temporaire supprimé")
        except Exception as e:
            logger.warning(f"⚠️ Erreur nettoyage: {e}")
    
    async def run_import_process(self) -> bool:
        """Processus complet d'import"""
        logger.info("📥 DÉMARRAGE IMPORT LLAMA3 8B DANS OLLAMA")
        
        # 1. Vérifications préalables
        if not self.check_model_file_exists():
            return False
        
        if not await self.check_ollama_available():
            return False
        
        # 2. Vérification si déjà chargé
        if await self.check_model_already_loaded():
            logger.info("✅ Modèle déjà disponible - import non nécessaire")
            return await self.test_imported_model()
        
        # 3. Création Modelfile
        modelfile_path = self.create_modelfile()
        
        try:
            # 4. Import modèle
            if not self.import_model_ollama(modelfile_path):
                return False
            
            # 5. Vérification import
            if not await self.check_model_already_loaded():
                logger.error("❌ Modèle non trouvé après import")
                return False
            
            # 6. Test fonctionnel
            return await self.test_imported_model()
            
        finally:
            # 7. Nettoyage
            self.cleanup_modelfile(modelfile_path)

async def main():
    """Import principal Llama3 8B"""
    try:
        importer = Llama3OllamaImporter()
        success = await importer.run_import_process()
        
        if success:
            print("\n🎊 IMPORT LLAMA3 8B RÉUSSI")
            print("🚀 Modèle prêt pour validation SuperWhisper V6")
            print(f"📋 Nom modèle: {importer.model_name}")
            print("🔄 Vous pouvez maintenant exécuter: python scripts/test_llama3_8b_validation.py")
            return 0
        else:
            print("\n❌ IMPORT LLAMA3 8B ÉCHOUÉ")
            print("🔧 Vérifiez Ollama CLI et fichier modèle")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Erreur import: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 