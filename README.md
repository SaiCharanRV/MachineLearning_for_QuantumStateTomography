# Quantum Density Matrix Reconstruction (Track 1)

This project implements a Machine Learning model to reconstruct quantum density matrices from Classical Shadows measurement data, enforcing physical constraints (Hermitian, PSD, Unit Trace).

**Track Selected**: Track 1 - Classical Shadows with Transformer Architecture.

## Repository Structure
- `src/`: Source code for data generation, model, training, and evaluation.
- `docs/`: Detailed documentation.
  - [Model Working](docs/model_working.md): Explanation of the architecture.
  - [Replication Guide](docs/replication_guide.md): Steps to run the code.
- `outputs/`: Directory for saved models and results.
- `AI_USAGE.md`: AI attribution policy.

> **Note on Hardware Simulation Files**: This implementation uses Track 1 (Classical Shadows + Transformer), which is a pure software approach using PyTorch. Hardware simulation files (`.vcd`) are not applicable to this track, as they require FPGA synthesis tools (Vivado, Vitis HLS) which are only relevant for Track 2 (Hardware-Centric implementations). The model weights are saved as `.pt` files in the `outputs/` directory.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python -m src.train
   ```
3. Evaluate:
   ```bash
   python -m src.evaluate
   ```

## Metrics
The evaluation script reports:
- Mean Fidelity (F)
- Trace Distance
- Inference Latency

See `FINAL_REPORT.md` for the detailed performance report and `docs/replication_guide.md` for usage instructions.
