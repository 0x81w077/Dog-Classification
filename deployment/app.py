# FastAPI service for the dog breed classifier (TensorFlow Lite model, or the Keras model when no .tflite file is present).
#
# Start:  uvicorn app:app --host 0.0.0.0 --port 8000
# Docs:   http://localhost:8000/docs
# Test:   curl -F "file=@dog.jpg" http://localhost:8000/predict
import io
import json
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

HERE = Path(__file__).parent
CLASS_NAMES = json.loads((HERE / 'class_names.json').read_text())
STORE = 256            # images were stored at 256x256 for training
UNSURE_BELOW = 0.35    # below this confidence the answer is flagged as unsure

if (HERE / 'model.tflite').exists():
    interpreter = tf.lite.Interpreter(model_path=str(HERE / 'model.tflite'))
    interpreter.allocate_tensors()
    IN = interpreter.get_input_details()[0]
    OUT = interpreter.get_output_details()[0]
    IMG = int(IN['shape'][1])

    def run_model(x):
        interpreter.set_tensor(IN['index'], x.astype(IN['dtype']))
        interpreter.invoke()
        return interpreter.get_tensor(OUT['index'])[0].astype(float)
else:
    import keras
    KERAS_MODEL = keras.models.load_model(HERE / 'model.keras')
    IMG = int(KERAS_MODEL.input_shape[1])

    def run_model(x):
        return KERAS_MODEL.predict(x, verbose=0)[0].astype(float)

lock = threading.Lock()        # one request at a time

app = FastAPI(title='Dog Breed Classifier', version='1.0')


def preprocess(im):
    # same steps as in training: fix rotation, RGB, short side -> 256, centre crop 256, centre crop 224
    im = ImageOps.exif_transpose(im).convert('RGB')
    w, h = im.size
    s = STORE / min(w, h)
    im = im.resize((max(STORE, round(w * s)), max(STORE, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - STORE) // 2, (h - STORE) // 2
    im = im.crop((l, t, l + STORE, t + STORE))
    off = (STORE - IMG) // 2
    im = im.crop((off, off, off + IMG, off + IMG))
    return np.asarray(im, dtype=np.float32)[None]


@app.get('/health')
def health():
    return {'status': 'ok', 'classes': len(CLASS_NAMES), 'input_size': IMG}


@app.post('/predict')
async def predict(file: UploadFile = File(...), top_k: int = 5):
    data = await file.read()
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        raise HTTPException(status_code=400, detail='The upload is not a readable image.')
    x = preprocess(im)
    with lock:
        p = run_model(x)
    top = np.argsort(-p)[:max(1, min(top_k, len(CLASS_NAMES)))]
    best = int(top[0])
    return {
        'filename': file.filename,
        'predicted_breed': CLASS_NAMES[best] if p[best] >= UNSURE_BELOW else 'unsure',
        'best_guess': CLASS_NAMES[best],
        'confidence': round(float(p[best]), 4),
        'top_k': [{'breed': CLASS_NAMES[int(i)], 'probability': round(float(p[int(i)]), 4)} for i in top],
    }
