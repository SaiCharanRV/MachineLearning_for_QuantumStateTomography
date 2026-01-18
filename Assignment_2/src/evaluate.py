import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
from tqdm import tqdm
import os

from src.data import QuantumStateDataset
from src.model import DensityMatrixReconstructor
from src.train import fidelity, trace_distance

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = DensityMatrixReconstructor(n_qubits=args.n_qubits).to(device)
    model_path = args.model_path if args.model_path else f"outputs/model_best_q{args.n_qubits}.pt"
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}. Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dataset (Test set)
    # Use a fixed seed for reproducibility of test set
    np.random.seed(42)
    test_dataset = QuantumStateDataset(n_qubits=args.n_qubits, n_samples=args.n_samples, n_shots=args.n_shots)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    fids = []
    tds = []
    latencies = []
    
    print("Starting Evaluation...")
    with torch.no_grad():
        for meas, outcomes, rho_target in tqdm(test_loader):
            meas = meas.to(device)
            outcomes = outcomes.to(device)
            rho_target = rho_target.to(device)
            
            # Measure Latency
            start_time = time.time()
            rho_pred = model(meas, outcomes)
            end_time = time.time()
            
            batch_time = end_time - start_time
            latencies.append(batch_time / meas.size(0)) # Per sample latency
            
            # Metrics
            fids.append(fidelity(rho_pred, rho_target))
            tds.append(trace_distance(rho_pred, rho_target))
            
    mean_fid = np.mean(fids)
    mean_td = np.mean(tds)
    mean_latency = np.mean(latencies) * 1000 # ms
    
    print("\nResults:")
    print(f"Mean Fidelity: {mean_fid:.4f}")
    print(f"Mean Trace Distance: {mean_td:.4f}")
    print(f"Avg Inference Latency: {mean_latency:.2f} ms/sample")
    
    # Save results
    with open("outputs/evaluation_report.txt", "w") as f:
        f.write(f"Mean Fidelity: {mean_fid:.4f}\n")
        f.write(f"Mean Trace Distance: {mean_td:.4f}\n")
        f.write(f"Avg Inference Latency: {mean_latency:.2f} ms/sample\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_qubits', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_shots', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()
    
    evaluate(args)
