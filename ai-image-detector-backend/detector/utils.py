# detector/utils.py
from pathlib import Path

def setup_project_structure():
    structure = {
        'dataset': {
            'real_images': 'Images réelles/authentiques',
            'ai_generated': 'Images générées par IA',
        },
        'models': 'Modèles entraînés sauvegardés',
        'results': 'Résultats d\'analyses et rapports',
        'test_images': 'Images à tester individuellement'
    }
    for folder, subfolders in structure.items():
        if isinstance(subfolders, dict):
            Path(folder).mkdir(exist_ok=True)
            for subfolder in subfolders:
                (Path(folder) / subfolder).mkdir(exist_ok=True)
        else:
            Path(folder).mkdir(exist_ok=True)
