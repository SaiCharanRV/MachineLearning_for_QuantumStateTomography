# Model Working: Density Matrix Reconstruction via Classical Shadows and Transformers

## 1. Problem Definition
The Quantum State Tomography (QST) problem asks: Given a set of measurement outcomes from an unknown quantum state $\rho$, how can we reconstruct the matrix $\rho$ such that it faithfully represents the state while satisfying the laws of quantum mechanics?

A physical density matrix $\rho$ for an $N$-qubit system ($d=2^N$) must satisfy three strict conditions:
1.  **Hermitian**: $\rho = \rho^\dagger$ (Self-adjoint).
2.  **Positive Semi-Definite (PSD)**: $\rho \succeq 0$ (All eigenvalues $\lambda_i \ge 0$, representing probabilities).
3.  **Unit Trace**: $\text{Tr}(\rho) = 1$ (Total probability is 1).

Standard linear inversion techniques often produce matrices that violate conditions 2 and 3 due to statistical noise (e.g., negative eigenvalues). Our Deep Learning approach solves this by **learning to predict a valid construction** rather than the matrix itself.

## 2. Methodology: Classical Shadows
We rely on the **Classical Shadows** protocol for data acquisition. Instead of full tomography which requires $O(d^2)$ measurements, we use random Pauli measurements.

### 2.1 The Measurement Protocol
1.  Select a random Pauli basis $P \in \{X, Y, Z\}$ for each qubit.
2.  Measure the qubit in this basis.
3.  Record the **Basis** (0, 1, 2) and the **Outcome** (+1, -1).

A single data point (snapshot) looks like:
$$ \text{Snapshot}_i = (\text{Basis}_i, \text{Outcome}_i) $$
For $N$ qubits, this is a sequence of length $N$.

## 3. Neural Architecture (Track 1)
We employ a **Transformer Encoder** architecture. Transformers are ideal because they process sets/sequences and capture correlations between qubits (entanglement) via the self-attention mechanism.

### 3.1 Input Embedding
The raw measurement data is embedded into a high-dimensional vector space:
-   **Basis Embedding**: Each basis choice ($X, Y, Z$) is mapped to a vector $\vec{e}_{basis} \in \mathbb{R}^{d_{model}}$.
-   **Outcome Embedding**: The outcome $b \in \{-1, 1\}$ is projected and added:
    $$ \vec{x}_{in} = \text{Embed}(P) + \text{Linear}(b) $$

### 3.2 Self-Attention Mechanism
The core logic resides in the Transformer layers:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
This allows the model to "attend" to correlations between different measurement shots or different qubits, effectively learning the structure of the quantum state from sparse data.

### 3.3 The Cholesky Head (Constraint Enforcement)
This is the most critical component. To ensure the output $\rho$ physically valid, the neural network does **not** predict $\rho$ directly. Instead, it predicts a lower-triangular matrix $L$ with complex entries.

1.  **Prediction**: Network outputs $2 \times d^2$ real numbers, interpreted as the real and imaginary parts of a $d \times d$ matrix $L_{raw}$.
2.  **Lower Triangular Projection**: We enforce $L_{ij} = 0$ for $j > i$.
3.  **Construction**:
    $$ \rho_{\text{unnorm}} = L L^\dagger $$
    *   **Why?** Any matrix of the form $A A^\dagger$ is automatically Hermitian and Positive Semi-Definite.
4.  **Normalization**:
    $$ \rho = \frac{\rho_{\text{unnorm}}}{\text{Tr}(\rho_{\text{unnorm}})} $$
    *   **Why?** This guarantees $\text{Tr}(\rho) = 1$.

## 4. Training Process
-   **Loss Function**: Mean Squared Error (MSE) between the predicted $\rho_{pred}$ and the true target $\rho_{true}$.
    $$ \mathcal{L} = \frac{1}{B} \sum_{i=1}^B || \rho_{pred}^{(i)} - \rho_{true}^{(i)} ||_F^2 $$
-   **Optimizer**: Adam ($\eta=10^{-3}$).
-   **Data**: Random mixed states generated from the Ginibre ensemble.

This end-to-end differentiable pipeline ensures that every gradient update moves the parameters towards generating physically valid states that align with the observed measurement data.
