# Plant Disease Prediction using Deep Learning 

This project utilizes a deep Convolutional Neural Network (CNN) to classify plant diseases from leaf images. It addresses the critical agricultural challenge of early disease detection, providing a fast, automated, and accessible diagnostic tool for farmers and gardeners. The model is trained on the "New Plant Diseases Dataset" and can classify leaves into 38 different categories with high accuracy.

## \#\# About The Project 

Plant diseases pose a significant threat to food security, leading to substantial crop losses worldwide. Manual disease detection is often slow, requires expert knowledge, and can be inconsistent. This project leverages the power of deep learning to automate this process.

The core of the project is a VGG-style CNN built with TensorFlow and Keras. After training on over 87,000 images, the model can predict the specific disease of a plant leaf from a new image, empowering users to take timely and appropriate action.

### \#\# Dataset

The model was trained on the **New Plant Diseases Dataset** available on Kaggle.

  * **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
  * **Content**: The dataset contains 87,000+ RGB images of plant leaves.
  * **Classes**: It is organized into 38 classes, each representing a specific plant and its health status (e.g., `Apple___Cedar_apple_rust`, `Corn_(maize)___healthy`).
  * **Structure**: The data is pre-split into `train` and `valid` directories, which is ideal for supervised learning.

-----

## \#\# Technology Stack 

This project was built using the following technologies and libraries:

  * **TensorFlow & Keras**: For building, training, and evaluating the deep learning model.
  * **Scikit-learn**: For generating detailed performance metrics like the classification report and confusion matrix.
  * **NumPy**: For numerical operations and handling image arrays.
  * **Matplotlib & Seaborn**: For data visualization, including plotting training history and the confusion matrix.
  * **Jupyter Notebook**: As the primary environment for development and experimentation.

-----

## \#\# Model Architecture 

The project uses a deep Sequential CNN architecture inspired by VGGNet. It consists of a convolutional base for feature extraction and a dense classifier head for prediction.

### \#\#\# Convolutional Base

The base is composed of five blocks of `Conv2D` and `MaxPool2D` layers. The number of filters doubles in each subsequent block, allowing the model to learn a hierarchy of features from simple edges to complex disease patterns.

```
Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool2D
Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool2D
Block 3: Conv2D(128) -> Conv2D(128) -> MaxPool2D
Block 4: Conv2D(256) -> Conv2D(256) -> MaxPool2D
Block 5: Conv2D(512) -> Conv2D(512) -> MaxPool2D
```

### \#\#\# Classifier Head

The classifier head takes the extracted features and makes the final prediction. Dropout layers are used to prevent overfitting.

```
-> Dropout(0.25)
-> Flatten()
-> Dense(1500, activation='relu')
-> Dropout(0.4)
-> Dense(38, activation='softmax')  // Output Layer
```

The model has over **7.8 million** trainable parameters.

-----

## \#\# Installation & Usage 

To get a local copy up and running, follow these steps.

1.  **Clone the repository**

    ```sh
    git clone https://github.com/your_username/your_repository_name.git
    ```

2.  **Install dependencies**
    It's recommended to create a virtual environment first.

    ```sh
    pip install tensorflow scikit-learn numpy matplotlib seaborn
    ```

3.  **Download the Dataset**

      * Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) provided above.
      * Unzip the file and ensure you have the `train` and `valid` directories in your project folder.

4.  **Train the Model**

      * Run the `Train_plant_disease.ipynb` notebook.
      * This will train the model for 10 epochs and save the trained model as `trained_plant_disease_model.keras`.

5.  **Test the Model**

      * Run the `Test_plant_disease.ipynb` notebook.
      * You can change the `image_path` variable in the notebook to test a prediction on any single leaf image from the `test` directory.

-----

## \#\# Results & Evaluation 

The model achieved an accuracy of approximately **95%** on the validation set, demonstrating strong generalization capabilities.

Performance was thoroughly analyzed using:

  * **Training vs. Validation Accuracy Plot**: To visually inspect the learning process and check for overfitting.

  * **Classification Report**: Provided detailed **Precision**, **Recall**, and **F1-Score** for each of the 38 classes.

  * **Confusion Matrix**: Offered a visual breakdown of where the model was making correct and incorrect predictions.

-----

## \#\# Future Improvements 

While the model performs well, there are several avenues for future improvement:

  * **Transfer Learning**: Implement a pre-trained model like **EfficientNet** or **ResNet50** to leverage knowledge from a larger dataset, which could further boost accuracy.
  * **Hyperparameter Tuning**: Use tools like KerasTuner to systematically find the optimal learning rate, dropout rate, and optimizer settings.
  * **Deployment**:
      * Wrap the model in a **Flask** or **FastAPI** web framework to create a REST API for online predictions.
      * Convert the model to **TensorFlow Lite** to deploy it in a mobile application for offline, real-time use by farmers in the field.
