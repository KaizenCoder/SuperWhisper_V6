#!/usr/bin/env python3
"""
🔧 CORRECTION AUTOMATIQUE LLM MANAGER
Corrige automatiquement l'API Ollama dans EnhancedLLMManager
🚨 CONFIGURATION GPU: RTX 3090 (CUDA:1) OBLIGATOIRE
"""

import os
import sys
import pathlib
import shutil
from datetime import datetime

# =============================================================================
# 🚨 CONFIGURATION CRITIQUE GPU - RTX 3090 UNIQUEMENT 
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        # RTX 3090 24GB EXCLUSIVEMENT
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ordre stable des GPU

print("🎮 GPU Configuration: RTX 3090 (CUDA:1) forcée")

# =============================================================================
# 🚀 PORTABILITÉ AUTOMATIQUE
# =============================================================================
def _setup_portable_environment():
    """Configure l'environnement pour exécution portable"""
    current_file = pathlib.Path(__file__).resolve()
    
    # Chercher le répertoire racine
    project_root = current_file
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'pyproject.toml', 'requirements.txt', '.taskmaster']):
            project_root = parent
            break
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.chdir(project_root)
    print(f"📁 Project Root: {project_root}")
    return project_root

_PROJECT_ROOT = _setup_portable_environment()

def create_backup(file_path):
    """Crée une sauvegarde du fichier"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"💾 Sauvegarde créée: {backup_path}")
    return backup_path

def fix_llm_manager():
    """Corrige le LLM Manager pour utiliser la bonne API Ollama"""
    
    llm_manager_path = "LLM/llm_manager_enhanced.py"
    
    if not os.path.exists(llm_manager_path):
        print(f"❌ Fichier non trouvé: {llm_manager_path}")
        return False
    
    # Créer sauvegarde
    backup_path = create_backup(llm_manager_path)
    
    try:
        # Lire le fichier actuel
        with open(llm_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Nouvelle méthode _generate_ollama corrigée
        new_generate_ollama = '''    async def _generate_ollama(self, user_input: str, max_tokens: int, temperature: float) -> str:
        """Génération via Ollama API - VERSION CORRIGÉE"""
        try:
            import httpx
            
            # ✅ CORRECTION: Utiliser l'API native Ollama avec le bon format
            actual_model = getattr(self, 'actual_model_name', self.config.get('model', 'nous-hermes'))
            
            # Format correct pour l'API native Ollama
            data = {
                "model": actual_model,
                "prompt": f"{self.system_prompt}\\n\\nUser: {user_input}\\nAssistant:",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["User:", "\\n\\n"]
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
            return None'''
        
        # Trouver et remplacer la méthode _generate_ollama
        start_marker = "    async def _generate_ollama(self, user_input: str, max_tokens: int, temperature: float) -> str:"
        end_marker = "            return None"
        
        start_index = content.find(start_marker)
        if start_index == -1:
            print("❌ Méthode _generate_ollama non trouvée")
            return False
        
        # Trouver la fin de la méthode (prochaine méthode ou fin de classe)
        lines = content[start_index:].split('\n')
        method_lines = [lines[0]]  # Première ligne (signature)
        
        for i, line in enumerate(lines[1:], 1):
            # Si on trouve une nouvelle méthode au même niveau d'indentation, on s'arrête
            if line.strip() and not line.startswith('    ') and not line.startswith('\t'):
                break
            if line.strip().startswith('def ') and line.startswith('    '):
                break
            method_lines.append(line)
            # Si on trouve la fin typique de la méthode
            if "return None" in line and line.strip().endswith("return None"):
                break
        
        old_method = '\n'.join(method_lines)
        end_index = start_index + len(old_method)
        
        # Remplacer la méthode
        new_content = content[:start_index] + new_generate_ollama + content[end_index:]
        
        # Écrire le fichier corrigé
        with open(llm_manager_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ LLM Manager corrigé avec succès!")
        print("🔧 Corrections appliquées:")
        print("  - API native Ollama (/api/generate)")
        print("  - Format de données correct")
        print("  - Gestion d'erreurs améliorée")
        print("  - Logs détaillés")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction: {e}")
        # Restaurer la sauvegarde en cas d'erreur
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, llm_manager_path)
            print(f"🔄 Fichier restauré depuis la sauvegarde")
        return False

def verify_fix():
    """Vérifie que la correction a été appliquée"""
    llm_manager_path = "LLM/llm_manager_enhanced.py"
    
    try:
        with open(llm_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier que la correction est présente
        checks = [
            "VERSION CORRIGÉE" in content,
            "/api/generate" in content,
            "num_predict" in content,
            "repeat_penalty" in content
        ]
        
        if all(checks):
            print("✅ Vérification: Correction appliquée avec succès")
            return True
        else:
            print("❌ Vérification: Correction incomplète")
            return False
            
    except Exception as e:
        print(f"❌ Erreur vérification: {e}")
        return False

def create_test_script():
    """Crée un script de test pour vérifier Ollama"""
    test_script = '''#!/usr/bin/env python3
"""
Test rapide Ollama après correction
"""

import asyncio
import sys
import os
import pathlib

# Setup portable
current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Configuration GPU RTX 3090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

async def test_corrected_llm():
    """Test du LLM Manager corrigé"""
    try:
        from LLM.llm_manager_enhanced import EnhancedLLMManager
        
        config = {
            'model': 'nous-hermes-2-mistral-7b-dpo:latest',
            'use_ollama': True,
            'timeout': 30.0
        }
        
        print("🧪 Test LLM Manager corrigé...")
        llm_manager = EnhancedLLMManager(config)
        await llm_manager.initialize()
        
        # Test simple
        response = await llm_manager.generate_response("Quelle est la capitale de la France ?")
        
        if response and response.strip():
            print(f"✅ Test réussi: {response}")
            return True
        else:
            print("❌ Test échoué: Pas de réponse")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_corrected_llm())
'''
    
    with open("test_ollama_corrected.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("📝 Script de test créé: test_ollama_corrected.py")

def main():
    """Fonction principale"""
    print("🔧 CORRECTION AUTOMATIQUE LLM MANAGER")
    print("=" * 50)
    
    print("\n📋 Étapes de correction:")
    print("1. Sauvegarde du fichier original")
    print("2. Application de la correction API Ollama")
    print("3. Vérification de la correction")
    print("4. Création d'un script de test")
    
    # Étape 1 & 2: Correction
    if fix_llm_manager():
        print("\n✅ Correction appliquée")
        
        # Étape 3: Vérification
        if verify_fix():
            print("✅ Vérification réussie")
            
            # Étape 4: Script de test
            create_test_script()
            
            print("\n🎯 PROCHAINES ÉTAPES:")
            print("1. Exécuter: python diagnostic_ollama_fix.py")
            print("2. Si Ollama fonctionne, tester: python test_ollama_corrected.py")
            print("3. Puis tester le pipeline complet: python test_pipeline_microphone_reel.py")
            
        else:
            print("❌ Vérification échouée")
    else:
        print("❌ Correction échouée")

if __name__ == "__main__":
    main() 