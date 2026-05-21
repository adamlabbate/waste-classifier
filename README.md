# Waste Classifier

A full stack web app that classifies waste images into 6 categories using a custom convolutional neural network trained from scratch with PyTorch.

## Results

**80.3% test accuracy** trained on 2,527 images over 100 epochs. No pretrained weights or transfer learning.

## Stack

- **ML model** — custom CNN built with PyTorch, trained on the [TrashNet](https://github.com/garythung/trashnet) dataset
- **Backend** — Flask REST API that runs inference and returns predictions as JSON
- **Frontend** — React (Vite) with drag-and-drop image upload and live results

## Waste Categories

Cardboard · Glass · Metal · Paper · Plastic · Trash

## Project Structure

```
src/
    model.py      # CNN architecture and class names
    train.py      # Training and validation loop
    predict.py    # Inference function
    utils.py      # Data loading and transforms
backend/
    app.py        # Flask server — POST /predict endpoint
frontend/
    src/
        App.jsx       # Root React component
        App.css       # Component styles
        index.css     # Global styles and CSS variables
        main.jsx      # App entry point
waste_classifier.ipynb  # Training notebook with experiments and results
requirements.txt
```

## Setup

1. **Clone the repo** and create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Install frontend dependencies:**

    ```bash
    cd frontend && npm install
    ```

## Running the app

Start the Flask backend from the project root:

```bash
python backend/app.py
```

In a separate terminal, start the React frontend:

```bash
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Training your own model

Place the TrashNet dataset at `data/TrashNet/dataset-resized` ([download here](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip)), then run:

```bash
python -m src.train
```

The script automatically uses GPU (CUDA or Apple MPS) if available.

## CLI prediction

To classify an image directly without the web app:

```bash
python -m src.predict --image path/to/image.jpg --model waste_classifier.pth
```
