# Predict the breed of one or more dog photos with the saved Keras model.
#
# Usage:  python prediction.py dog1.jpg dog2.jpg
#         python prediction.py dog.jpg --model ../models/transfer_learning_model.keras --classes ../models/class_names.json --top 5
import argparse
import json
from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageOps

STORE = 256            # images were stored at 256x256 for training
UNSURE_BELOW = 0.35    # below this confidence the answer is flagged as unsure


def load_image(path, img):
    # load -> fix rotation -> RGB -> short side 256 -> centre crop 256 -> centre crop to the network size
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    w, h = im.size
    s = STORE / min(w, h)
    im = im.resize((max(STORE, round(w * s)), max(STORE, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - STORE) // 2, (h - STORE) // 2
    im = im.crop((l, t, l + STORE, t + STORE))
    off = (STORE - img) // 2
    return np.asarray(im.crop((off, off, off + img, off + img)), dtype=np.float32)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--model', default=str(here.parent / 'models' / 'transfer_learning_model.keras'))
    ap.add_argument('--classes', default=str(here.parent / 'models' / 'class_names.json'))
    ap.add_argument('--top', type=int, default=5)
    a = ap.parse_args()
    model = keras.models.load_model(a.model)
    names = json.loads(Path(a.classes).read_text())
    img = model.input_shape[1]
    for path in a.images:
        p = model.predict(load_image(path, img)[None], verbose=0)[0]
        top = np.argsort(-p)[:a.top]
        print(f'Image: {Path(path).name}')
        print(f'Predicted Breed: {names[top[0]] if p[top[0]] >= UNSURE_BELOW else "unsure (mixed breed or not a dog?)"}')
        print(f'Confidence: {p[top[0]]:.2%}')
        print('Top guesses: ' + ', '.join(f'{names[i]} {p[i]:.1%}' for i in top))
        print()


if __name__ == '__main__':
    main()
