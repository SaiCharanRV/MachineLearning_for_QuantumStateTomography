# Replication Guide

## 1. Environment Setup
To reproduce the results, you need a standard Python environment with PyTorch.

### 1.1 Prerequisites
-   Python 3.8 or higher.
-   `pip` package manager.

### 1.2 Installation
1.  **Clone the repository** (or extract the project folder):
    ```bash
    cd paac_ass-2
    ```
2.  **Install Dependencies**:
    The project relies on `torch` (for the model), `numpy` & `scipy` (for math), and `tqdm` (for progress bars).
    ```bash
    pip install -r requirements.txt
    ```

## 2. Dataset Generation
The training pipeline generates data **on-the-fly** to ensure infinite variety and prevent overfitting to a fixed dataset.
-   **Logic**: `src/data.py` contains the `QuantumStateDataset` class.
-   **Process**: It generates random density matrices (Ginibre ensemble) and simulates Pauli measurements (using the Born rule).

You do **not** need to run a separate script to create data files. Training handles this automatically.

## 3. Training the Model
We provide a unified training script `src/train.py`.

### 3.1 Replication Command (Best Result)
To replicate the **High Fidelity (>0.97)** result for single-qubit tomography:

```bash
python -m src.train --n_qubits 1 --n_samples 5000 --epochs 30 --lr 0.001
```

### 3.2 Explanation of Arguments
-   `--n_qubits 1`: Specifies Single Qubit Tomography (Track 1 target).
-   `--n_samples 5000`: Generates 5,000 unique quantum states for training.
-   `--epochs 30`: Iterates over the dataset 30 times.
-   `--n_shots 100`: Simulates 100 measurement shots per state.

### 3.3 Training Output
The script will display a progress bar and save the best model to `outputs/`:
```text
Epoch 1/30: 100%|██████| ... loss: 0.152
...
Validation - Fidelity: 0.9775
Training Complete. Best Fidelity: 0.9775
```
Artifact produced: `outputs/model_best_q1.pt`

## 4. Evaluation & Validation
Once trained, we use **two** validation methods.

### 4.1 Standard ML Evaluation
Evaluates the model on a new, unseen test set of 1000 random states.
```bash
python -m src.evaluate --n_qubits 1 --model_path outputs/model_best_q1.pt
```
**Expected Output**:
-   Mean Fidelity: **~0.977**
-   Trace Distance: **~0.11**
-   Latency: **<1ms**

### 4.2 Assignment 1 "Bridge" Validation
Validates the model against specific reference states (Bell States / Basis States) to compare with Linear Inversion.
```bash
python -m src.validate_assignment_1 --model_path outputs/model_best_q1.pt --n_qubits 1
```

## 5. Troubleshooting
-   **Low Fidelity?**: Ensure you are using `n_qubits=1`. For `n_qubits=2`, significantly more training (N=100k) is required.
-   **Out of Memory?**: Reduce batch size: `--batch_size 16`.
-   **Import Errors?**: Make sure you run commands from the root folder `paac_ass-2` as `python -m src.script`.
