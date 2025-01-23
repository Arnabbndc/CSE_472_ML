# CSE_472_ML

## Overview

This repository contains assignments for the CSE 472 Machine Learning course. Each assignment focuses on different machine learning techniques and algorithms, implemented using Python and Jupyter notebooks.

## Assignments

### Assignment 1: Data Preprocessing

**Objective**: Clean, transform, and prepare the HR Employee Attrition dataset for analysis.

**Key Features**:

- **Handling Missing Values**: Identifying and imputing missing values to ensure data completeness.
- **Encoding Categorical Variables**: Converting categorical variables into numerical format using techniques like Label Encoding.
- **Feature Scaling**: Standardizing and normalizing numerical features to bring them to a common scale.
- **Removing Duplicates**: Identifying and removing duplicate rows to maintain data integrity.
- **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships between features to gain insights.

### Assignment 2: Logistic Regression with Bagging and Stacking

**Objective**: Implement logistic regression models with bagging and stacking techniques to improve model performance.

**Key Features**:

- **Bagging (Bootstrap Aggregating)**: Training multiple logistic regression models on different bootstrap samples and aggregating their predictions to reduce variance and improve accuracy.
- **Stacking**: Training multiple logistic regression models as base learners and using a meta-learner to combine their predictions, enhancing model performance by leveraging the strengths of individual models.

### Assignment 3: Feed-Forward Neural Network (FNN)

**Objective**: Build and train a Feed-Forward Neural Network for a given dataset.

**Key Features**:

- **Network Architecture Design**: Designing the architecture of the FNN, including the number of layers, neurons, and activation functions (e.g., ReLU, Sigmoid).
- **Model Training**: Compiling the model with appropriate loss functions and optimizers, and training it on the dataset while monitoring performance using validation data.
- **Performance Evaluation**: Evaluating the model's performance using metrics such as accuracy, precision, recall, and F1-score.
- **Model Saving**: Saving the best model configuration and weights for future use.

### Assignment 4: PCA & EM Algorithm

**Objective**: Implement Principal Component Analysis (PCA) for dimensionality reduction and the Expectation-Maximization (EM) algorithm for clustering.

**Key Features**:

- **Principal Component Analysis (PCA)**:
  - **Data Standardization**: Standardizing the data to have a mean of zero and a standard deviation of one.
  - **Covariance Matrix Calculation**: Computing the covariance matrix to understand the variance between features.
  - **Eigenvalues and Eigenvectors**: Calculating eigenvalues and eigenvectors to identify the principal components.
  - **Dimensionality Reduction**: Transforming the data into a new subspace with reduced dimensions while retaining most of the variance.
  - **Visualization**: Visualizing the transformed data in the new subspace to understand the impact of dimensionality reduction.

- **Expectation-Maximization (EM) Algorithm**:
  - **Initialization**: Initializing parameters such as means, covariances, and mixing coefficients.
  - **E-step (Expectation)**: Computing the responsibilities, which are the probabilities that each data point belongs to each cluster.
  - **M-step (Maximization)**: Updating the parameters based on the computed responsibilities.
  - **Iteration**: Repeating the E-step and M-step until convergence to find the optimal clustering of the data.


### ML PROJECT: [Deep Fake Image Detector Robust Against Adversarial Attack](https://github.com/Arnabbndc/CSE-472-ML-Project)

[Visit the project repository for more details](https://github.com/Arnabbndc/CSE-472-ML-Project)

- **Computer Vision** Python, PyTorch, YOLOv11
  
Developed a deep fake image detector robust against adversarial attacks using the CiFake Dataset

- **Black-Box Experiments**: Evaluated models (ViT, YOLOv11x) on noisy datasets; applied feature dropping and
augmentations, achieving up to 80% accuracy on adversarial data

- **White-Box Experiments**: Enhanced model robustness with custom Adversarial Noise layers and Dropout Layers


## How to Run

1. Navigate to the respective assignment folder.
2. Follow the instructions in the README file of each assignment to run the code.

## Conclusion

This repository demonstrates various machine learning techniques and algorithms, including data preprocessing, logistic regression with ensemble methods, neural networks, PCA, and the EM algorithm. Each assignment provides a practical implementation and evaluation of these techniques.

## References

- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [NumPy Documentation](https://numpy.org/doc/)

---
