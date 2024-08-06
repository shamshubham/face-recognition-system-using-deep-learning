# 🌟 Face Recognition System with Deep Learning 🌟

Welcome to the **Face Recognition System** project, where cutting-edge deep learning meets computer vision! This project showcases a complete pipeline for face recognition, from dataset generation to model training and prediction visualization. Leveraging powerful libraries like OpenCV and tflearn, it provides a robust and intuitive system for recognizing faces.

## 📚 Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🌟 Introduction

This project captures images via a webcam, detects faces, and compiles a dataset. Using a Convolutional Neural Network (CNN) trained on this dataset, the system can recognize various individuals. The model classifies images into one of three categories based on its training.

## 🚀 Getting Started

### Prerequisites

Before diving in, ensure you have the following tools and libraries installed:

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

   Install all required packages:

   ```bash
   pip install opencv-python numpy tflearn tensorflow matplotlib tqdm
   ```

3. **Download Haar Cascade for Face Detection**:

   Obtain the `haarcascade_frontalface_default.xml` file and place it in the project directory.

## 🎉 Usage

1. **Generate Dataset**:

   Capture images and build your dataset by running:

   ```bash
   python FaceRecognitionSystem_using_Deep_learning.ipynb
   ```

2. **Train the Model**:

   Once the dataset is ready, the script trains a CNN model on the collected data.

3. **Visualize Predictions**:

   The script also includes tools to visualize predictions made by the trained model on a separate set of images.

## 🔍 Code Overview

### 1. Dataset Generation

- **Function**: `generate_dataset()`
- **Purpose**: Captures images using a webcam, detects faces with a Haar Cascade classifier, and saves cropped grayscale face images.

### 2. Data Preparation

- **Function**: `my_data()`
- **Purpose**: Reads images, assigns labels, and preprocesses data for model training.

### 3. Model Training

- **Model Architecture**:

  - Convolutional layers with ReLU activation
  - Max pooling layers
  - Fully connected layer with ReLU activation
  - Output layer with softmax activation

- **Training**: Utilizes categorical crossentropy loss and the Adam optimizer.

### 4. Visualization

- **Function**: `data_for_visualization()`
- **Purpose**: Visualizes model predictions on a new set of images.

## 🛠 Technologies Used

- **OpenCV**: Face detection and image processing.
- **NumPy**: Efficient numerical operations.
- **tflearn**: Building and training the CNN model.
- **TensorFlow**: Backend for deep learning operations.
- **Matplotlib**: Visualizing predictions and data.
- **tqdm**: Displaying progress bars during data processing.

## 🤝 Contributing

Contributions are always welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## 💡 Acknowledgments

- Special thanks to the OpenCV community for providing comprehensive tools for computer vision.
- Gratitude to the creators of tflearn and TensorFlow for enabling deep learning innovation.
- Appreciation to the authors of the Haar Cascade XML files for facilitating face detection.
