# Waste Classifier - Project Context

## What this is
A full stack waste image classifier. The ML model is built and trained. 
Now building a web app on top of it.

## ML Model (complete)
- Custom CNN built from scratch in PyTorch
- Trained on TrashNet dataset (6 classes: cardboard, glass, metal, paper, plastic, trash)
- 80% test accuracy
- predict() function in src/predict.py takes an image path and returns a class name

## File structure
waste-classifier/
├── src/
│   ├── model.py       # WasteClassifier CNN architecture
│   ├── utils.py       # data loading and transforms
│   ├── train.py       # training loop
│   └── predict.py     # inference function + CLI script
├── backend/           # not built yet - needs Flask server
├── frontend/          # React app (Vite), partially set up
│   └── src/
│       ├── App.jsx    # root component, cleaned up and ready
│       └── main.jsx   # entry point
├── waste_classifier.ipynb
└── requirements.txt

## What to build next
1. Flask backend (backend/app.py) with one POST endpoint /predict that:
   - Accepts an image file
   - Runs it through src/predict.py
   - Returns the predicted class as JSON

2. React frontend with three components:
   - Upload area (drag and drop or click to upload)
   - Image preview
   - Result display

## Preferences
- Walk me through what you're doing, don't just generate code silently
- Explain concepts I might not know
- I have HTML/CSS/JS experience but React is relatively new to me