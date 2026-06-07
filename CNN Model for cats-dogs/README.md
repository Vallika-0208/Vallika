Dogs vs Cats Image Classification Using CNN

Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images as either Dog or Cat. The model is trained on a Dogs vs Cats dataset with data augmentation techniques to improve performance and reduce overfitting.

Features

Image classification using CNN
Data augmentation for better generalization
Batch Normalization and Dropout layers
Early Stopping and Learning Rate Reduction
Validation Accuracy and Loss Curves
Confusion Matrix Evaluation
Prediction on new images

Technologies Used

Python
TensorFlow / Keras
NumPy
Matplotlib
Seaborn
Scikit-learn
Dataset

##Dataset used: Dogs vs Cats Dataset##

Source: Kaggle

Download the dataset from Kaggle and extract it into the project directory before training.

Expected folder structure:

extracted_data/
│
├── train/
│   ├── cat/
│   └── dog/
│
└── val/
    ├── cat/
    └── dog/
##Project Structure##
Dogs-vs-Cats-Classification/
│
├── train_model.py
├── test_model.py
├── requirements.txt
├── training_validation_curves.png
├── confusion_matrix.png
├── best_cnn_model.h5
└── README.md
##Model Architecture##

The CNN architecture contains:

Conv2D Layer (32 Filters)
Batch Normalization
Max Pooling
Dropout
Conv2D Layer (64 Filters)
Batch Normalization
Max Pooling
Dropout
Conv2D Layer (128 Filters)
Batch Normalization
Max Pooling
Dropout
Dense Layer (512 Units)
Dropout
Output Layer (Sigmoid)
Training

##Run the following command:##

python train_model.py

The training script:

Loads training and validation data
Applies data augmentation
Trains the CNN model
Generates validation curves
Generates confusion matrix
Saves the trained model as:
best_cnn_model.h5
Testing

Run the following command:

python test_model.py

The testing script:

Loads the trained model
Loads a test image
Performs prediction
Displays Cat/Dog classification with confidence score
Results

The project evaluates model performance using:

Validation Accuracy Curve
Validation Loss Curve
Confusion Matrix
Classification Report

Generated files:

training_validation_curves.png
confusion_matrix.png
Requirements

Install dependencies:

pip install tensorflow numpy matplotlib seaborn scikit-learn

Or use:

pip install -r requirements.txt
Future Improvements
Transfer Learning using MobileNetV2
ResNet50 Integration
EfficientNet Models
Web-based Prediction Interface
Real-time Image Classification
Author

Sajja Rangavallika

B.Tech – Artificial Intelligence and Data Science

Dogs vs Cats Classification Project using TensorFlow and CNN.
