import torch
import torch.nn as nn
import numpy as np

class DensityMatrixReconstructor(nn.Module):
    def __init__(self, n_qubits=2, d_model=64, n_heads=2, n_layers=2, dim_feedforward=128):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        
        # Embeddings
        # 3 Pauli bases (0,1,2) -> X,Y,Z
        self.basis_embedding = nn.Embedding(3, d_model)
        
        # Outcome processing: Map scalar outcome to d_model
        # We can concatenate outcome to embedding or multiply.
        # Let's simple projection of outcome (-1, 1).
        self.outcome_embedding = nn.Linear(1, d_model)
        
        # Position encoding is NOT needed because measurements are permutation invariant (set),
        # but Transformers assume sequence. We can use a learnable "measurement token" or just no pos encoding.
        # Given "Classical Shadows" are essentially a set of snapshots, permutation invariance is key.
        # So no positional encoding is physically motivated.
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Head to predict L (lower triangular)
        # We output full matrix and mask it, or just n*(n+1)/2 elements?
        # Outputting full real+imag parts of dim x dim matrix is easiest to implement.
        output_dim = 2 * (self.dim ** 2)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, meas, outcomes):
        """
        Args:
            meas: (batch, n_shots, n_qubits) Indices 0-2
            outcomes: (batch, n_shots, 1) Values -1, 1
        """
        batch_size = meas.size(0)
        n_shots = meas.size(1)
        
        # Embed measurements: (batch, n_shots, n_qubits, d_model) -> sum over qubits to get shot embedding?
        # Or flatten qubits into sequence? 
        # Track 1 suggests Transformer.
        # Option A: Each shot is a token. Input dim = n_qubits * basis_dim + 1?
        # Option B: Embed each qubit's basis then sum/concat.
        
        # Let's sum embeddings of basis per shot.
        # meas_embed: (batch, n_shots, n_qubits, d_model)
        meas_embed = self.basis_embedding(meas) 
        # Sum over qubits to get representation of the Pauli string used in that shot
        shot_basis_embed = meas_embed.sum(dim=2)  # (batch, n_shots, d_model)
        
        # Encode outcome
        outcome_embed = self.outcome_embedding(outcomes) # (batch, n_shots, d_model)
        
        # Combine: Elementwise add (or concat)
        x = shot_basis_embed + outcome_embed
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (batch, n_shots+1, d_model)
        
        # Transform
        x = self.transformer(x)
        
        # Pool: Take CLS token output
        x_pool = x[:, 0, :]
        
        # Predict params
        out = self.head(x_pool) # (batch, 2 * dim^2)
        
        # Reshape to complex matrix
        out = out.view(batch_size, self.dim, self.dim, 2)
        L_real = out[..., 0]
        L_imag = out[..., 1]
        L_raw = torch.complex(L_real, L_imag)
        
        # Make Lower Triangular
        L = torch.tril(L_raw)
        
        # Enforce positive diagonal elements for uniqueness of Cholesky (not strictly required for density, 
        # but standard). Actually, L L^\dagger is unique only if diag is real > 0.
        # But for just generating a valid rho, any L works.
        
        # 2. Reconstruct Rho: rho = L L^\dagger / tr(L L^\dagger)
        # L_dag = L.conj().transpose(-2, -1)
        # rho_unnorm = L @ L_dag
        # Use einsum for batch matmul to be safe
        rho_unnorm = torch.matmul(L, L.conj().transpose(-2, -1))
        
        trace = torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1) # (batch,)
        trace = trace.real # Trace of Hermitian is real
        
        # Avoid div by zero (shouldn't happen with random init but safe guard)
        rho_recon = rho_unnorm / (trace.view(-1, 1, 1).unsqueeze(-1) + 1e-8)
        # Fix shape broadcasting for complex division if needed, but PyTorch handles it.
        # Actually trace is real, rho is complex.
        rho_recon = rho_unnorm / (trace.view(-1, 1, 1) + 1e-8)
        
        return rho_recon

if __name__ == "__main__":
    # Test model
    model = DensityMatrixReconstructor(n_qubits=2)
    # Fake batch
    m = torch.randint(0, 3, (2, 10, 2))
    o = torch.randn(2, 10, 1)
    rho = model(m, o)
    print("Output shape:", rho.shape)
    print("Trace:", torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1))
    print("Hermitian check:", torch.allclose(rho, rho.conj().transpose(-2, -1), atol=1e-5))
