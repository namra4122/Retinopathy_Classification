# Diabetic Retinopathy Classification

This project implements various deep learning models to classify diabetic retinopathy severity using retinal images. The models include VGG16, VGG19, Xception, and InceptionV3, all pre-trained on ImageNet and fine-tuned for this specific task.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Performance Evaluation](#performance-evaluation)
7. [Usage](#usage)
8. [Results](#results)
9. [Future Work](#future-work)

## Project Overview

Diabetic retinopathy is a diabetes complication that affects the eyes. Early detection is crucial for preventing vision loss. This project aims to automate the classification of diabetic retinopathy severity using machine learning techniques on retinal images.

## Prerequisites

The project requires the following libraries:

- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Pillow

You can install these dependencies using pip:

```
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow
```

## Dataset

The dataset should be organized in the following structure:

```
Diabetic Retinopathy ML Dataset/
├── train/
│   ├── class_0/
│   ├── class_1/
│   ├── class_2/
│   ├── class_3/
│   └── class_4/
└── test/
    ├── class_0/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    └── class_4/
```

Each class represents a severity level of diabetic retinopathy.

## Model Architecture

The project implements four different models:

1. VGG16
2. VGG19
3. Xception
4. InceptionV3

Each model is pre-trained on ImageNet and fine-tuned for diabetic retinopathy classification. The models are modified by:

- Removing the top layers
- Adding custom fully connected layers
- Setting specific layers to be trainable or non-trainable

## Training Process

The training process includes:

1. Data preprocessing and augmentation
2. Splitting data into train, validation, and test sets
3. Model compilation with SGD optimizer and categorical crossentropy loss
4. Training for a specified number of epochs with early stopping

Key hyperparameters:

- Image dimensions: 176x208
- Batch size: 16-64
- Learning rate: 0.0001
- Momentum: 0.9
- Epochs: 30 (adjustable)

## Performance Evaluation

The models are evaluated on three sets:

1. Training set
2. Validation set
3. Test set

Metrics used:

- Accuracy
- Loss

## Usage

To train and evaluate a model:

1. Prepare your dataset in the required directory structure.
2. Adjust the hyperparameters in the script if needed.
3. Run the script for the desired model (VGG16, VGG19, Xception, or InceptionV3).
4. The trained model will be saved in HDF5 format.

## Results

The performance of each model is printed after training, showing the accuracy on the train, validation, and test sets.

## Future Work

Potential improvements and extensions:

1. Implement k-fold cross-validation
2. Experiment with other architectures (e.g., ResNet, DenseNet)
3. Implement ensemble methods
4. Analyze model interpretability using techniques like Grad-CAM
5. Deploy the best performing model as a web service

Feel free to contribute to this project by submitting pull requests or opening issues for bugs and feature requests.
