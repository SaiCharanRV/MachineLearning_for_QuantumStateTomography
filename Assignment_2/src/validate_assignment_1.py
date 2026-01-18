import torch
import numpy as np
import argparse
import os
from src.model import DensityMatrixReconstructor
from src.train import fidelity, trace_distance
# Reuse data generation tools
from src.data import simulate_measurements, get_pauli_string, PAULI_BASIS
import scipy.linalg

# ==========================================
# 0. Linear Inversion (Baseline)
# ==========================================

def linear_inversion_reconstruction(meas_np, outcomes_np):
    """
    Reconstruct density matrix using Linear Inversion.
    rho_rec = sum_i (outcome_i * P_i) / N_shots (approx, need frame op)
    For Pauli measurements:
    rho = 1/2^N sum_{P} <P> P
    """
    n_shots, n_qubits = meas_np.shape
    dim = 2**n_qubits
    
    # Calculate expectation values for each Pauli string
    # Map for easy lookup?
    # Simple approach: Accumulate P * outcome
    rho_accum = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n_shots):
        basis_indices = meas_np[i] # [0, 1] etc
        outcome = outcomes_np[i]
        
        # Construct P
        P = np.array([1.0+0j])
        for idx in basis_indices:
            P = np.kron(P, PAULI_BASIS[idx])
            
        # Linear inversion formula for Pauli basis:
        # rho = 1/d I + sum_{P!=I} <P> P / d
        # But here we are sampling uniformly from 3^N bases.
        # Standard CS formula: rho_hat = (2^N+1)*U|b><b|U^dagger - I
        # Let's use the Standard CS formula if possible, but our data is Pauli.
        # Yes, Pauli basis IS the CS basis.
        # Shadow snapshot: 3 \otimes ... \otimes 3 U^dag |b><b| U - I
        # For single qubit Pauli X measured with outcome +1:
        # State collapsed to |+>, so snapshot is 3|+><+| - I.
        
        # Let's simple Linear Inversion of <P>:
        # rho = 1/2^N * sum_{P in {I,X,Y,Z}^N} <P> P
        # We need to estimate <P>.
        
        # We have single shot.
        # Let's use the "Classical Shadow" snapshot formula for each shot, then average.
        # Snapshot_i = big_tensor_product( 3 * P_projector_i - I )
        
        snapshot = np.array([1.0+0j])
        for q in range(n_qubits):
            basis_idx = basis_indices[q] # 0,1,2
            measure_op = PAULI_BASIS[basis_idx]
            
            # Outcome +1 -> eigenvec corresponding to +1
            # Outcome -1 -> eigenvec corresponding to -1
            evals, evecs = np.linalg.eigh(measure_op)
            # evecs columns are eigenvectors. 
            # If outcome is +1 (index 1 in sorted evals? No, likely index 1 is +1, index 0 is -1 for Z)
            # X: evals -1, 1. 
            
            # Map outcome to index
            # -1 -> 0, 1 -> 1
            if outcome == -1: # Wait, outcome is for the WHOLE string parity? 
                # No, data.py simulate_measurements returns outcome +/-1 for the WHOLE string trace.
                # It does NOT return per-qubit outcomes.
                # This makes CS reconstruction harder without per-qubit outcomes.
                pass
            
            # WAIT. simulate_measurements in data.py returns `outcome` for the global trace Tr(P rho).
            # It does NOT emulate "measure each qubit in X, Y, Z and get bitstring".
            # It picks a GLOBAL Pauli string P, calculates <P>, and flips coin.
            # So, we are estimating <P>.
            
            # Estimator for <P>: outcome_i (if we measured P).
            # We measured P_i.
            
            # So, we sum (outcome_i * P_i) for all P_i, and divide by (total_count_of_P_i / total_shots)?
            # No.
            # We are sampling P uniformly from 3^N strings.
            # d = 2^N.
            # Estimator of rho:
            # rho = (1/d) * sum_{P} Tr(rho P) P
            # We measure P_i, get o_i approx Tr(rho P_i).
            # So estimator is:
            # rho ~ (1/d) * (1/N_shots) * sum_i (outcome_i * P_i) * (Total_Num_Paulis)
            # Total Paulis = 3^N (ignoring Identity).
            # So, rho_rec = (3^N / d) * mean(outcome_i * P_i)
            pass

    # Correct formula given the sampling:
    # We choose P uniformly from 3^N basis strings.
    # E[outcome * P] = sum_{Q} prob(Q measured) * Tr(rho Q) * Q
    #                = sum_{Q} (1/3^N) * Tr(rho Q) * Q
    # We want sum_{Q} Tr(rho Q) Q / 2^N
    # So we need to multiply average by 3^N / 2^N.
    # rho_rec = (3/2)^N * mean(outcome_i * P_i)
    # Plus we need the Identity component which is not measured?
    # Tr(rho I) = 1.
    # The sum usually includes I. Our sampling doesn't measure I.
    # So rho = I/2^N + (sum_{P!=I} <P>P)/2^N.
    # Our samples estimate the second part.
    
    snapshot_sum = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n_shots):
        basis_indices = meas_np[i] 
        outcome = outcomes_np[i]
        
        P = np.array([1.0+0j])
        for idx in basis_indices:
            P = np.kron(P, PAULI_BASIS[idx])
            
        snapshot_sum += outcome * P
        
    mean_snapshot = snapshot_sum / n_shots
    
    # rho_rec = I/d + (3^N / d) * mean_snapshot
    rho_rec = np.eye(dim)/dim + ((3**n_qubits)/dim) * mean_snapshot
    
    # Force PSD? Linear inversion doesn't force PSD.
    # But usually we project to PSD.
    return rho_rec

# ==========================================
# 1. Define Assignment 1 Reference States
# ==========================================

def get_single_qubit_states():
    """
    Returns a dictionary of single-qubit density matrices from Assignment 1.
    """
    states = {
        "|0>": np.array([1, 0], dtype=complex),
        "|1>": np.array([0, 1], dtype=complex),
        "|+>": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "|->": np.array([1, -1], dtype=complex) / np.sqrt(2),
        "(|0>+i|1>)/sqrt(2)": np.array([1, 1j], dtype=complex) / np.sqrt(2)
    }
    
    dms = {}
    for name, psi in states.items():
        psi = psi.reshape(-1, 1)
        rho = psi @ psi.conj().T
        # Add dummy second qubit for 2-qubit model (tensor with |0>)
        # Or should we just test on 1-qubit model?
        # The user's prompt implies scalable workflow. 
        # But if we trained a 2-qubit model, we should test on 2-qubit states.
        # However, the user explicitly asked to "bridge Ass 1". Ass 1 had single qubit.
        # Let's tensor with |0> to make them compatible with a 2-qubit model, 
        # OR train a 1-qubit model as well. Best approach: Test 2-qubit Bell States (Ass 1 Task 3).
        dms[name] = rho
        
    return dms

def get_bell_states():
    """
    Returns the 4 Bell states for 2-qubit validation.
    """
    bell_vectors = {
        "Phi+": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        "Phi-": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        "Psi+": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        "Psi-": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    }
    
    dms = {}
    for name, psi in bell_vectors.items():
        psi = psi.reshape(-1, 1)
        rho = psi @ psi.conj().T
        dms[name] = rho
    return dms

# ==========================================
# 2. Validation Loop
# ==========================================

def validate_assignment_1(model_path, n_qubits=2, n_shots=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load Model
    model = DensityMatrixReconstructor(n_qubits=n_qubits).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Validating on Assignment 1 States (N={n_qubits})")
    print(f"{'='*60}")
    
    states_to_test = {}
    
    if n_qubits == 2:
        states_to_test.update(get_bell_states())
        print("Included States: Bell States (Phi+, Phi-, Psi+, Psi-)")
    else:
        # If we trained a 1-qubit model
        states_to_test.update(get_single_qubit_states())
        print("Included States: Single Qubit Reference States")

    results = []
    
    print(f"\n{'State':<20} | {'Method':<10} | {'Fidelity':<10}")
    print("-" * 50)
    
    with torch.no_grad():
        for name, rho_true in states_to_test.items():
            # 1. Simulate Experiment (Generate Shots)
            meas_np, outcomes_np = simulate_measurements(rho_true, n_shots)
            
            # --- Method A: ML Model ---
            meas = torch.tensor(meas_np, dtype=torch.long).unsqueeze(0).to(device)
            outcomes = torch.tensor(outcomes_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
            rho_pred_ml = model(meas, outcomes)
            rho_pred_np_ml = rho_pred_ml.squeeze(0)
            
            rho_true_tensor = torch.tensor(rho_true, dtype=torch.complex64).to(device)
            fid_ml = fidelity(rho_pred_ml, rho_true_tensor.unsqueeze(0))
            
            # --- Method B: Linear Inversion (Baseline) ---
            # Linear Inversion on the SAME data
            rho_rec_li = linear_inversion_reconstruction(meas_np, outcomes_np)
            
            # Compute Fidelity for LI
            # Need to convert to torch manually
            rho_rec_li_t = torch.tensor(rho_rec_li, dtype=torch.complex64).to(device).unsqueeze(0)
            fid_li = fidelity(rho_rec_li_t, rho_true_tensor.unsqueeze(0))
            
            results.append((name, fid_ml, fid_li))
            print(f"{name:<20} | ML         | {fid_ml:.4f}")
            print(f"{'':<20} | Lin. Inv.  | {fid_li:.4f}")
            print("-" * 50)

    # Summary
    avg_fid_ml = np.mean([r[1] for r in results])
    avg_fid_li = np.mean([r[2] for r in results])
    
    print(f"\nAverage Fidelity (ML): {avg_fid_ml:.4f}")
    print(f"Average Fidelity (LI): {avg_fid_li:.4f}")
    
    # Save to file
    with open(f"outputs/validation_ass1_q{n_qubits}_comparison.txt", "w") as f:
        f.write(f"Comparative Validation (N={n_qubits})\n")
        f.write("-" * 40 + "\n")
        for name, fml, fli in results:
            f.write(f"{name}: ML={fml:.4f}, LI={fli:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Avg ML: {avg_fid_ml:.4f}\n")
        f.write(f"Avg LI: {avg_fid_li:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--n_qubits', type=int, default=2, help='Number of qubits')
    parser.add_argument('--n_shots', type=int, default=1000, help='Number of measurement shots')
    args = parser.parse_args()
    
    validate_assignment_1(args.model_path, args.n_qubits, args.n_shots)
