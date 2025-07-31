# Human Activity Recognition with Multi-Modal Sensor Fusion (Centralized and Federated Approaches) [Time-MAGNET]

This project explores two different approaches – **Centralized Training** and **Federated Learning** – for classifying human activities using data from multiple sensor modalities: Accelerometer (Thigh), Accelerometer (Wrist), Depth Camera, and Pressure Mat.

## Project Goal

The primary objective is to build a robust activity recognition system by effectively fusing information from diverse sensor streams. We compare a traditional centralized training paradigm, where all data is aggregated, with a federated learning approach, where models are trained locally on client devices and updates are aggregated to a global model, preserving data privacy.

## Data

The project utilizes a multi-modal dataset containing sensor readings collected from participants performing various activities. The data includes:

*   **Accelerometer (Thigh and Wrist):** Time series data representing acceleration along x, y, and z axes.
*   **Depth Camera:** Image-like data representing depth information.
*   **Pressure Mat:** Image-like data representing pressure distribution.

The data is preprocessed to reformat timestamps, add headers, split into overlapping windows, resample to fixed lengths, normalize, and augment.

## Model Architecture

Both the centralized and federated approaches employ a shared model architecture designed for multi-modal fusion:

*   **Modality Encoders:**
    *   **Accelerometer (Thigh and Wrist):** Customized T5 transformer Encoder with LoRA for efficient time series feature extraction.
    *   **Depth Camera and Pressure Mat:** Custom CNN Encoder with attention mechanisms for spatial feature extraction.
*   **Fusion Module:** An Enhanced Fusion Module that utilizes Graph Attention, Mixture of Experts (MoE), and SwiGLU layers to learn complex interactions between the different sensor modalities. It supports dynamic and learnable adjacency to model relationships between modalities.
*   **Classifier:** A multi-layer perceptron with GELU activation, BatchNorm, and Dropout for classifying the fused features into activity classes.

## Training Approaches

### Centralized Training

In this approach, all preprocessed sensor data from all participants is combined into a single large dataset. A single instance of the multi-modal fusion model is trained directly on this aggregated dataset using standard backpropagation.

### Federated Learning

This approach simulates a federated learning scenario. The data from each participant is treated as belonging to a separate client.

1.  **Client Datasets:** The training data is partitioned by participant, creating individual datasets for each client.
2.  **Global Model:** A global model is initialized.
3.  **Federated Rounds:** The training proceeds in rounds. In each round:
    *   A subset of clients is selected.
    *   Each selected client downloads the current global model.
    *   Each client trains the model locally on their private data for a few epochs.
    *   Clients send their model *updates* (gradients or model differences) back to a central server.
    *   The server aggregates the updates from the selected clients (e.g., using Federated Averaging) to produce a new global model.
4.  This iterative process allows the global model to learn from the distributed data without the raw data ever leaving the clients.

## Evaluation

Both approaches are evaluated on a separate test set using standard classification metrics, including:

*   Accuracy
*   Weighted F1-score
*   Precision
*   Recall
*   Confusion Matrix
*   Classification Report

Visualizations such as loss curves, accuracy curves, t-SNE plots, UMAP plots, ROC-AUC curves, and Precision-Recall curves are generated to analyze the performance and learned representations.

## Implementation Details

The models are implemented using PyTorch. The customized T5 encoder leverages the `transformers` and `peft` libraries. Training includes techniques like gradient accumulation, mixed precision training (`autocast`, `GradScaler`), and learning rate scheduling. The federated learning implementation includes client selection and model aggregation logic.
