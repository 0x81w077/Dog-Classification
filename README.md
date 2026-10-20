# Dog Breed Classification Using Deep Learning (CNN and Transfer Learning)
Notebook : `https://colab.research.google.com/drive/1Cog_jf8QOKI-MtsYthnOycdCPif6FRDJ?usp=sharing`

## What this project does
Classifies a photo of a dog into one of 138 breeds. A CNN built from scratch is compared with a transfer-learning model (EfficientNetV2B0, ImageNet weights, tuned with Optuna).

## Results (test set, 5560 unseen images)
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Custom CNN | 52.6% | 48.2% | 43.6% | 43.5% |
| Transfer learning, default | 83.9% | 84.9% | 83.7% | 83.7% |
| Transfer learning, tuned | 84.2% | 85.0% | 83.7% | 83.8% |

## Folder structure
```
Dog_Image_Classification/
  dataset/        how to obtain the data (not included: too large)
  notebooks/      dog_classification.ipynb (run in Google Colab, T4 GPU)
  models/         cnn_model, transfer_learning_model (.keras, .h5, .tflite), class_names.json
  test_images/    unseen photos used for the prediction tests
  results/        accuracy/loss curves, confusion matrix, classification report, predictions, comparison tables
  src/            prediction.py (command-line prediction)
  deployment/     app.py (FastAPI service using the TensorFlow Lite model)
  report/         project report
  requirements.txt
```

## Dataset
See `dataset/README.md` for names, links and how to get them (Kaggle token needed for the Kaggle sets).

## How to run
1. Open the notebook in Google Colab, choose Runtime > Change runtime type > T4 GPU, then Runtime > Run all. Progress is saved to Google Drive.
2. Predict with the saved model: `pip install -r requirements.txt` then `python src/prediction.py test_images/<photo>.jpg`
3. Run the web service: `cd deployment && pip install -r requirements.txt && uvicorn app:app --port 8000`, then open http://localhost:8000/docs or `curl -F "file=@dog.jpg" http://localhost:8000/predict`

## Notes
* Duplicates were removed before the train/validation/test split, so no photo is in two sets.
* Stanford Dogs and the Kaggle competition set are subsets of ImageNet, which the pretrained network has seen, so scores on those photos are optimistic.
