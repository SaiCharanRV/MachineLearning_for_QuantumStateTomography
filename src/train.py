import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm

from src.data import QuantumStateDataset
from src.model import DensityMatrixReconstructor

def fidelity(rho, sigma):
    """
    Compute quantum fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2.
    Since rho and sigma are density matrices (Hermitian, PSD), we can use:
    If one is pure: F = <psi|sigma|psi>
    General case: Use scipy or torch matrix sqrt.
    
    For training monitoring, we can approximate or just use this on CPU for validation.
    """
    # Using sqrtm is expensive on GPU in batch. 
    # For 2-qubit (4x4), it is fine.
    # Let's do it on CPU batch-wise or loop.
    
    rho = rho.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    
    fids = []
    from scipy.linalg import sqrtm
    
    for i in range(rho.shape[0]):
        r = rho[i]
        s = sigma[i]
        
        # sqrt(rho)
        sqrt_r = sqrtm(r)
        
        target = sqrt_r @ s @ sqrt_r
        root_target = sqrtm(target)
        
        fid = np.real(np.trace(root_target)) ** 2
        fids.append(np.clip(fid, 0, 1))
        
    return np.mean(fids)

def trace_distance(rho, sigma):
    """0.5 * Tr(|rho - sigma|)"""
    rho = rho.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    dists = []
    for i in range(rho.shape[0]):
        diff = rho[i] - sigma[i]
        # Eigenvalues of diff
        evals = np.linalg.eigvalsh(diff)
        td = 0.5 * np.sum(np.abs(evals))
        dists.append(td)
    return np.mean(dists)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    train_dataset = QuantumStateDataset(n_qubits=args.n_qubits, n_samples=args.n_samples, n_shots=args.n_shots)
    val_dataset = QuantumStateDataset(n_qubits=args.n_qubits, n_samples=args.n_samples // 10, n_shots=args.n_shots)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = DensityMatrixReconstructor(n_qubits=args.n_qubits).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss: MSE between rho_pred and rho_true
    # Since they are complex, MSE = mean(|rho_pred - rho_true|^2)
    criterion = nn.MSELoss()
    
    best_val_fid = 0.0
    
    os.makedirs('outputs', exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for meas, outcomes, rho_target in pbar:
            meas = meas.to(device)
            outcomes = outcomes.to(device)
            rho_target = rho_target.to(device)
            
            optimizer.zero_grad()
            
            rho_pred = model(meas, outcomes)
            
            # Complex MSE: Need to view as real to use torch MSELoss properly or split
            # torch.view_as_real is good
            loss = criterion(torch.view_as_real(rho_pred), torch.view_as_real(rho_target))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        val_fids = []
        val_tds = []
        with torch.no_grad():
            for meas, outcomes, rho_target in val_loader:
                meas = meas.to(device)
                outcomes = outcomes.to(device)
                rho_target = rho_target.to(device)
                
                rho_pred = model(meas, outcomes)
                
                fid = fidelity(rho_pred, rho_target)
                td = trace_distance(rho_pred, rho_target)
                val_fids.append(fid)
                val_tds.append(td)
                
        avg_fid = np.mean(val_fids)
        avg_td = np.mean(val_tds)
        print(f"Validation - Fidelity: {avg_fid:.4f}, Trace Dist: {avg_td:.4f}")
        
        if avg_fid > best_val_fid:
            best_val_fid = avg_fid
            torch.save(model.state_dict(), f"outputs/model_best_q{args.n_qubits}.pt")
            
    print(f"Training Complete. Best Fidelity: {best_val_fid:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_qubits', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_shots', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)
