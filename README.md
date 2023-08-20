# Text Emotion Classification using Machine Learning and Keras

This repository contains code and analysis for text emotion classification using Machine Learning and Keras. The project includes data preprocessing, tokenization, one-hot encoding, model architecture definition, training, and emotion prediction.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Emotion Prediction](#emotion-prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to classify the emotions in text using a Machine Learning model built with the Keras framework. The project showcases the steps involved in tokenizing text, one-hot encoding labels, defining a neural network architecture, training the model, and predicting emotions in input text.

## Prerequisites

### clone the repository : 

```bash
git clone https://github.com/Mohshaikh23/Text-Emotions-Classification.git
```

Before running the code, ensure you have the required libraries installed:

- pandas
- numpy
- keras
- tensorflow
- scikit-learn

You can install them using the following command:

```bash
pip install pandas numpy keras tensorflow scikit-learn
```

## Data Preprocessing

The code starts with loading and preprocessing text data. It tokenizes the text, pads sequences, encodes labels, and prepares the data for model training.

## Model Architecture

The neural network model architecture is defined using the Keras `Sequential` API. It consists of an embedding layer, a flatten layer, and two dense layers with specified activation functions.

## Model Training

The model is compiled with an optimizer and loss function, then trained using the prepared training data. Validation data is used to monitor the model's performance during training.

## Emotion Prediction

The trained model is used to predict emotions in input text. An example input text is preprocessed, and the model predicts the corresponding emotion label.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.