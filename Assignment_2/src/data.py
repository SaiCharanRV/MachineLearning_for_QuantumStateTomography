import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.linalg

# Pauli Matrices
I = np.eye(2, dtype=np.complex64)
X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
PAULI_BASIS = [X, Y, Z]

def random_density_matrix(n_qubits):
    """Generate a random density matrix using the Ginibre ensemble."""
    dim = 2**n_qubits
    # Random complex matrix
    G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    # Form rho = G G^dagger
    rho = G @ G.conj().T
    # Normalize trace
    rho /= np.trace(rho)
    return rho.astype(np.complex64)

def get_pauli_string(pauli_indices, n_qubits):
    """Convert list of indices (0=X, 1=Y, 2=Z) to full tensor product matrix."""
    mat = np.array([1.0+0j])
    for idx in pauli_indices:
        mat = np.kron(mat, PAULI_BASIS[idx])
    return mat

def simulate_measurements(rho, n_shots):
    """
    Simulate randomized Pauli measurements.
    For each shot:
    1. Pick a random Pauli basis P for each qubit.
    2. Measure rho in basis P = P_1 \otimes ... \otimes P_n.
    3. Outcome is +/- 1.
    
    Returns:
        measurements: (n_shots, n_qubits) integers in [0, 2] representing basis (X,Y,Z).
        outcomes: (n_shots,) integers {-1, 1} (eigenvalue).
    """
    n_qubits = int(np.log2(rho.shape[0]))
    measurements = np.random.randint(0, 3, size=(n_shots, n_qubits))
    outcomes = []

    for i in range(n_shots):
        basis_indices = measurements[i]
        P = get_pauli_string(basis_indices, n_qubits)
        
        # Probability of measuring +1: Tr(P rho) is trace of observable mean? 
        # No, Prob(+1) = Tr(M+ rho) where M+ is projector onto +1 eigenspace of P.
        # P has evals +1 and -1. P = (+1)M+ + (-1)M-.
        # Exp val <P> = Tr(P rho). 
        # Prob(+1) = (1 + <P>)/2, Prob(-1) = (1 - <P>)/2.
        
        exp_val = np.real(np.trace(P @ rho))
        # Clip for numerical stability
        exp_val = np.clip(exp_val, -1.0, 1.0)
        p_plus = 0.5 * (1 + exp_val)
        
        outcome = np.random.choice([1, -1], p=[p_plus, 1 - p_plus])
        outcomes.append(outcome)

    return measurements, np.array(outcomes)

class QuantumStateDataset(Dataset):
    def __init__(self, n_qubits=2, n_samples=1000, n_shots=100):
        self.n_qubits = n_qubits
        self.n_samples = n_samples
        self.n_shots = n_shots
        self.data = []
        
        print(f"Generating {n_samples} random density matrices for {n_qubits} qubits...")
        for _ in range(n_samples):
            rho = random_density_matrix(n_qubits)
            meas, outcomes = simulate_measurements(rho, n_shots)
            # Store flattened real/imag parts of rho for supervised target
            # rho is (2^N, 2^N) complex. 
            # We'll return measurement data and target rho.
            self.data.append({
                'measurements': meas, # (n_shots, n_qubits)
                'outcomes': outcomes, # (n_shots,)
                'rho': rho
            })

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        # Inputs: Flatten measurements and outcomes for the model
        # Or keep separate? Let's stack them.
        # Model input: (n_shots, n_qubits + 1) where last col is outcome?
        # Or (n_shots, n_qubits) basis indices and (n_shots, 1) outcomes.
        
        meas = torch.tensor(item['measurements'], dtype=torch.long)
        out = torch.tensor(item['outcomes'], dtype=torch.float32).unsqueeze(1)
        
        # Target rho: (2^N, 2^N) complex
        # PyTorch doesn't support complex perfectly in all optimizers/losses sometimes, 
        # but let's stick to complex tensors or split real/imag. 
        # Let's split for safety in NN output, but target can be complex.
        rho = torch.tensor(item['rho'], dtype=torch.complex64)
        
        return meas, out, rho
    
if __name__ == "__main__":
    # Test generation
    ds = QuantumStateDataset(n_qubits=2, n_samples=5, n_shots=10)
    m, o, r = ds[0]
    print("Measurements shape:", m.shape)
    print("Outcomes shape:", o.shape)
    print("Rho shape:", r.shape)
