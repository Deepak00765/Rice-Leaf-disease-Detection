Rice Leaf Disease Detection
Overview
This project focuses on detecting rice leaf diseases using a Convolutional Neural Network (CNN). The dataset consists of 119 images divided into three classes:

Bacterial Leaf Blight

Brown Spot

Leaf Smut

Due to the small dataset size, data augmentation is used to generate additional images and balance the data. The VGG19 model, a pre-trained deep learning architecture, is fine-tuned for this task to achieve high accuracy in disease classification.

Project Structure
Copy
rice-leaf-disease-detection/
├── data/
│   ├── train/
│   │   ├── Bacterial_Leaf_Blight/
│   │   ├── Brown_Spot/
│   │   └── Leaf_Smut/
│   ├── validation/
│   │   ├── Bacterial_Leaf_Blight/
│   │   ├── Brown_Spot/
│   │   └── Leaf_Smut/
│   └── test/
│       ├── Bacterial_Leaf_Blight/
│       ├── Brown_Spot/
│       └── Leaf_Smut/
├── models/
│   └── vgg19_rice_leaf_model.h5
├── scripts/
│   ├── data_augmentation.py
│   ├── train_model.py
│   └── evaluate_model.py
├── requirements.txt
└── README.md
Dataset
The dataset contains 119 images of rice leaves categorized into three classes:

Bacterial Leaf Blight

Brown Spot

Leaf Smut

To address the small dataset size, data augmentation is applied to generate additional images and balance the classes.

Data Augmentation
Data augmentation is performed using TensorFlow's ImageDataGenerator. The following transformations are applied:

Rotation

Flipping (horizontal and vertical)

Zooming

Shifting (width and height)

Shearing

This helps increase the dataset size and improve the model's ability to generalize.

Model Architecture
The VGG19 model is used for this project. VGG19 is a pre-trained CNN model with 19 layers, trained on the ImageNet dataset. The model is fine-tuned for rice leaf disease detection by:

Freezing the base layers of VGG19.

Adding custom layers (e.g., Dense, Dropout) on top.

Training the model on the augmented dataset.

Training
The model is trained using the following steps:

Data Preparation:

Split the dataset into training, validation, and test sets.

Apply data augmentation to the training set.

Model Compilation:

Use the Adam optimizer with a learning rate of 0.0001.

Use categorical_crossentropy as the loss function.

Apply class weights to handle class imbalance.

Training:

Train the model for 20 epochs with early stopping to prevent overfitting.

Evaluation
The model is evaluated on the test set using metrics like accuracy, precision, recall, and F1-score. A confusion matrix is generated to analyze misclassifications.

Requirements
To run this project, install the required libraries:

bash
Copy
pip install -r requirements.txt
Usage
Data Augmentation:

bash
Copy
python scripts/data_augmentation.py
Train the Model:

bash
Copy
python scripts/train_model.py
Evaluate the Model:

bash
Copy
python scripts/evaluate_model.py
Results
Test Accuracy: 92.8


Future Work
Collect more data to improve model performance.

Experiment with other pre-trained models like ResNet or EfficientNet.

Deploy the model as a web or mobile application for real-time disease detection.
