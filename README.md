# Waste Classifier

A deep learning project for classifying waste images into 6 categories (cardboard, glass, metal, paper, plastic, and trash) using a convolutional neural network built with PyTorch.

## Features

- Custom CNN model for image classification
- Training, validation, and testing pipeline
- Image prediction script for new images
- Utilities for data loading and preprocessing

## Waste Categories

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Results

Achieved **80.3% test accuracy** trained from scratch on 2,527 images over 100 epochs. No pretrained weights or transfer learning were used.

## Project Structure

```
src/
	model.py      # CNN model definition and class names
	train.py      # Training/validation loop
	predict.py    # Image prediction script
	utils.py      # Data loading and preprocessing utilities
data/
	# Place your dataset here (expects TrashNet dataset)
waste_classifier.ipynb  # training notebook with experimentation and results
requirements.txt
README.md
```

## Setup

1. **Clone the repository** and navigate to the project folder.

2. **Install dependencies** (preferably in a virtual environment):

	 ```
	 pip install -r requirements.txt
	 ```

3. **Prepare the dataset**  
	 Place your dataset in `data/TrashNet/dataset-resized` following the [TrashNet](https://github.com/garythung/trashnet) folder structure.

## Training

To train the model:

```
python src/train.py
```

The script will automatically use GPU if available.

## Prediction

To classify a new image:

```
python src/predict.py --image path/to/image.jpg --model path/to/model.pth
```

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Pillow
- Numpy

(See `requirements.txt` for full list.)

## Future Improvements

- Add batch normalization to stabilize training
- Add a learning rate scheduler to reduce val accuracy bouncing in later epochs
- Balance class distribution (paper: 594 images vs trash: 137 images)
- Deploy as a full stack web app with a React frontend and Flask backend