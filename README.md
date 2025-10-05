# Sports Image Classification using InceptionResNetV2

This notebook demonstrates the process of building an image classification model to classify different sports using the InceptionResNetV2 transfer learning model from Keras Applications.

## Problem Description

The goal is to build a model that can accurately identify various sports from images. This involves:
- Loading and exploring a dataset of sports images.
- Preprocessing the images for model training.
- Building a convolutional neural network using a pre-trained model (InceptionResNetV2) and fine-tuning it for the specific task.
- Training the model.
- Evaluating the model's performance.
- Visualizing model predictions and using Grad-CAM to understand which parts of the image influence the model's decision.

## Dataset

The dataset used is the "sports-classification" dataset from Kaggle, downloaded using `kagglehub`. The dataset is divided into training, validation, and test sets, with various sports categories.

The dataset structure is:
- `train/`: Training images for each sport.
- `valid/`: Validation images for each sport.
- `test/`: Test images for each sport.

Each sport is represented by a folder containing images belonging to that sport.

## Data Preprocessing and Augmentation

Images are loaded and resized to a uniform size of (224, 224). The datasets are created using `keras.utils.image_dataset_from_directory` with `label_mode="categorical"` for multi-class classification.

Data augmentation is applied to the training set to enhance the model's robustness and generalization capabilities. The augmentation layers include:
- `RandomFlip("horizontal_and_vertical")`
- `RandomRotation(0.2)`
- `RandomZoom(0.2)`
- `RandomTranslation(0.1, 0.1)`

## Model Architecture

The model utilizes the pre-trained InceptionResNetV2 model as its backbone, with weights trained on the ImageNet dataset. The `include_top` is set to `False` to remove the original classification layer, allowing us to add our own. The backbone is set to be trainable for fine-tuning.

A new classification head is added on top of the InceptionResNetV2 backbone:
- A `Flatten` layer to flatten the output of the convolutional layers.
- A `Dense` layer with 1024 units and ReLU activation.
- A `Dropout` layer with a rate of 0.25 for regularization.
- A final `Dense` layer with 100 units (equal to the number of sports classes) and softmax activation for multi-class classification.

The model is compiled with the Adam optimizer with a learning rate of 1e-5 and `CategoricalCrossentropy` loss.

## Training

The model is trained for a specified number of epochs with an EarlyStopping callback to prevent overfitting. The EarlyStopping monitors the validation loss and stops training if there is no significant improvement for a certain number of epochs (`patience=5`). A ModelCheckpoint callback is also used to save the model at the end of each epoch.

The training uses the augmented training dataset (`augmented_train_ds`) and the validation dataset (`evaluation_ds`).

## Evaluation

The model's performance is evaluated on the test dataset (`test_ds`) using:
- **Accuracy Score:** The overall percentage of correctly classified images.
- **Classification Report:** Provides precision, recall, and f1-score for each class.
- **Confusion Matrix:** A heatmap visualizing the true versus predicted labels for each class.

## Prediction

The notebook includes an example of how to load an image, preprocess it, and use the trained model to predict the sport.

## Grad-CAM Visualization

Grad-CAM is used to visualize the parts of the image that the model focuses on when making a prediction. This helps in understanding the model's decision-making process. The implementation involves:
- Getting the output of the last convolutional layer of the backbone.
- Calculating the gradients of the predicted class score with respect to the last convolutional layer's output.
- Generating a heatmap based on the gradients and superimposing it on the original image.

This visualization helps confirm whether the model is examining relevant features in the image for classification.

## Author

* **Juan Guillermo Gómez**
* Linkedin: [@jggomezt](https://www.linkedin.com/in/jggomezt/)

## License

    Copyright 2025 Juan Guillermo Gómez

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS

**[Aquí puedes añadir un resumen de los resultados que obtuviste]**

  * **Precisión (Accuracy) en el conjunto de validación:** [Ej: 95%]
  * **Matriz de confusión:** [Puedes pegar una imagen de la matriz de confusión aquí]

## 👨‍💻 Autor

  * **[Tu Nombre]**
