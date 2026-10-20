# Dog breed API

```
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to try it in the browser, or:

```
curl -F "file=@dog.jpg" http://localhost:8000/predict
```
