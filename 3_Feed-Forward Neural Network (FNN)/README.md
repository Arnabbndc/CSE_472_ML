# Assignment 3: Feed-Forward Neural Network (FNN)

## Overview

This assignment focuses on building and training a Feed-Forward Neural Network (FNN) for a given dataset. The goal is to design the network architecture, train the model, and evaluate its performance.

## Directory Structure

```
CSE_472_ML/
    3_Feed-Forward Neural Network (FNN)/
        1905065_best_model_config.pkl
        1905065_best_model_weights.pkl
        1905065.ipynb
```

## Files

- **1905065.ipynb**: Jupyter notebook containing the implementation and training of the Feed-Forward Neural Network.
- **1905065_best_model_config.pkl**: Pickle file containing the configuration of the best model.
- **1905065_best_model_weights.pkl**: Pickle file containing the weights of the best model.

## Dataset
<!-- https://www.kaggle.com/datasets/zalando-research/fashionmnist -->
The dataset used for this assignment is the [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset, which consists of 60,000 training images and 10,000 testing images. Each image is a 28x28 grayscale image associated with a label from 10 classes. The dataset is available in the Keras library.
Labels:

0. T-shirt/top  
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot


## Feed-Forward Neural Network (FNN)

1. **Network Architecture**:
    - Design the architecture of the FNN, including the number of layers and neurons in each layer.
    - Use activation functions such as ReLU and Sigmoid.

2. **Training the Model**:
    - Compile the model with an appropriate loss function and optimizer.
    - Train the model on the training dataset, using validation data to monitor performance.

3. **Evaluation**:
    - Evaluate the model's performance on the test dataset using metrics such as accuracy, precision, recall, and F1-score.
    - Save the best model configuration and weights.

## How to Run

Follow instructions in the [`1905065_Report.pdf`](./1905065_Report.pdf) file to run the code and visualize the results.


## Results

- The performance of the FNN is evaluated using various metrics.
- The best model configuration and weights are saved for future use.

## Conclusion

This assignment demonstrates the implementation and training of a Feed-Forward Neural Network. The model's performance is evaluated, and the best configuration and weights are saved for future use.

## References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---
