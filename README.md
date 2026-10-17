# CKD Prediction System

A Machine Learning-based web app that predicts Chronic Kidney Disease (CKD) from patient clinical attributes.

## Stack
- Python / Flask
- scikit-learn (Random Forest Classifier)
- Gunicorn (production server)

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## Deployment

This app is configured for deployment on [Render](https://render.com):
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn app:app`
