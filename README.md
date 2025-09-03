# Neural Radiance Fields (NeRF) for Image Generation

This project implements a Neural Radiance Field (NeRF) to generate an image. It uses two Multi-Layer Perceptrons (MLPs): a standard MLP and a Fourier Feature MLP. The goal is to learn a continuous representation of an image from a set of input coordinates and their corresponding pixel values.

## Features

-   **Image Loading and Preprocessing:** Loads an image, converts it to a PyTorch tensor, and crops it to a 512x512 resolution.
-   **Data Generation:** Creates training and testing datasets by mapping pixel coordinates to their RGB values.
-   **Two MLP Architectures:**
    -   **Feature MLP:** A standard MLP with ReLU activation functions and a final sigmoid activation.
    -   **Fourier Feature MLP:** An MLP that incorporates Fourier features to better capture high-frequency details in the image.
-   **Model Training:** Trains both MLPs using the Adam optimizer and Mean Squared Error (MSE) loss.
-   **Performance Evaluation:** Measures the performance of the models using the Peak Signal-to-Noise Ratio (PSNR).
-   **Visualization:** Plots the training loss and PSNR values over epochs, and displays the generated image from the best-performing model.
-   **Optimal Scale Search:** For the Fourier Feature MLP, it searches for the optimal scale for the Gaussian Fourier Features.

## Requirements

-   Python 3
-   PyTorch
-   TorchVision
-   Matplotlib
-   NumPy

You can install the required Python libraries using pip:

```bash
pip install torch torchvision matplotlib numpy
