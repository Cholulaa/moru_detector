import cv2
import numpy as np
from pathlib import Path
import sys

def simulate_windows_screenshot(img):
    # Simule un screenshot Windows : légère compression JPEG, pas de bruit ni de modification
    _, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 98])  # 98 = quasi lossless
    img_jpg = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    return img_jpg

def process_folder(input_folder, output_folder, extensions=['.jpg','.jpeg','.png','.bmp']):
    in_folder = Path(input_folder)
    out_folder = Path(output_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    files = [f for f in in_folder.glob('*') if f.suffix.lower() in extensions]
    print(f"Traitement de {len(files)} images depuis {input_folder}")
    for f in files:
        try:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                screened = simulate_windows_screenshot(img)
                out_target = out_folder / f.name
                cv2.imwrite(str(out_target), screened)
                print(f"✓ {f.name} → {out_target}")
        except Exception as e:
            print(f"Erreur {f.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simulate_win_screen.py dossier_source dossier_cible")
        exit(1)
    input_folder, output_folder = sys.argv[1], sys.argv[2]
    process_folder(input_folder, output_folder)
