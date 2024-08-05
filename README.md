# Face Recognition System

This project implements a face recognition system using deep learning techniques. It includes dataset generation, model training, and visualization of predictions. The system uses OpenCV for face detection and tflearn for building and training the CNN model.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The project captures images using a webcam, detects faces, and saves them for creating a dataset. A Convolutional Neural Network (CNN) is trained on this dataset to recognize different individuals. The system can classify images into one of three categories based on the trained model.

## Getting Started

### Prerequisites

- [Python](https://www.python.org/downloads/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [tflearn](http://tflearn.org/)
- [TensorFlow](https://www.tensorflow.org/)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/shamshubham/face-recognition-system-using-deep-learning.git
   cd face-recognition-system
   ```

2. **Install Dependencies**:

   Install the required packages using pip:

   ```bash
   pip install opencv-python numpy tflearn tensorflow matplotlib tqdm
   ```

3. **Download Haar Cascade for Face Detection**:

   Download the `haarcascade_frontalface_default.xml` file and place it in the project directory.

## Usage

1. **Generate Dataset**:

   To capture images and generate the dataset, run:

   ```bash
   python FaceRecognitionSystem_using_Deep learning.ipynb
   ```

   Replace `script_name.py` with the actual name of your Python script file.

2. **Train the Model**:

   After generating the dataset, the script will train a CNN model on the collected data.

3. **Visualize Predictions**:

   The script also includes functionality to visualize predictions made by the trained model on a separate set of images.

## Code Overview

### 1. Dataset Generation

The `generate_dataset()` function captures images from a webcam, detects faces using a Haar Cascade classifier, and saves the cropped face images in grayscale format.

### 2. Data Preparation

The `my_data()` function reads the images, assigns labels, and preprocesses the data for training.

### 3. Model Training

A CNN is defined and trained using tflearn with the following layers:

- Convolutional layers with ReLU activation
- Max pooling layers
- Fully connected layer with ReLU activation
- Output layer with softmax activation

The model is trained with categorical crossentropy loss and Adam optimizer.

### 4. Visualization

The `data_for_visualization()` function and subsequent code visualize the model's predictions on a separate set of images.

## Technologies Used

- **OpenCV**: For face detection and image processing.
- **NumPy**: For numerical operations.
- **tflearn**: For building and training the CNN model.
- **TensorFlow**: Backend for tflearn.
- **Matplotlib**: For visualizing predictions.
- **tqdm**: For displaying progress bars during data processing.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with your improvements.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The OpenCV library for providing powerful computer vision tools.
- The tflearn and TensorFlow communities for excellent deep learning resources.
- The authors and maintainers of the Haar Cascade XML files for face detection.
