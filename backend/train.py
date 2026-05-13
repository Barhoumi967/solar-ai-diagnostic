"""
Point d'entrée d'entraînement — Solar AI Diagnostic.

Lance : python backend/train.py  (depuis la racine du projet)
ou    : python train.py           (depuis le répertoire backend/)

Délègue à backend/models/train_all.py qui contient le pipeline complet.
"""

import sys
from pathlib import Path

# Assure que le répertoire backend est dans le path Python
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.train_all import main

if __name__ == "__main__":
    main()
