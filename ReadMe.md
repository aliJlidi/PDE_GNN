# How This Implements “PDE → GNN”

This repository demonstrates how a Graph Neural Network (GNN) layer can be interpreted as a discrete approximation of a diffusion–reaction Partial Differential Equation (PDE). The key ideas are:

1. **Diffusion term**: smooth node features via the (normalized) graph Laplacian.  
2. **Reaction term**: apply a learnable linear transform (followed by ReLU) to each node’s features.  
3. **Combine** both to update node embeddings in one layer.

---

## Diffusion Term

\[
\text{diffusion} \;=\; \alpha \bigl(\tilde{A}\,x \;-\; x\bigr)
\]

- \(\tilde{A}x\) is the normalized adjacency–feature multiplication.  
- \(\alpha\) is the diffusion coefficient (controls smoothing strength).

---

## Reaction Term

\[
\text{reaction} \;=\; \beta \, \sigma\bigl(W\,x\bigr)
\]

- \(W\) is a learnable weight matrix.  
- \(\sigma(\cdot)\) is ReLU.  
- \(\beta\) is the reaction coefficient (controls feature update strength).

---

## Overall Update

Combining diffusion and reaction gives the per-layer update:

\[
x^{(l+1)} \;=\; x^{(l)} 
\;+\; \alpha \bigl(\tilde{A}\,x^{(l)} - x^{(l)}\bigr)
\;+\; \beta \,\sigma\bigl(W^{(l)}\,x^{(l)}\bigr).
\]

This matches the LaTeX formula:

\[
H^{(l+1)} = H^{(l)} 
\;+\; \alpha \bigl(\tilde{A}\,H^{(l)} - H^{(l)}\bigr) 
\;+\; \beta\,\sigma\bigl(H^{(l)}\,W^{(l)}\bigr).
\]

---

## Repository Structure

- `pde_gnn_link_pred.py`  
  - Implements both a **baseline GCN** and a **PDE‐inspired GNN** for link prediction.  
  - Contains training loops, evaluation, and plotting of ROC curves (Cora or PubMed dataset).  

- `README.md`  
  - Explains the PDE‐to‐GNN connection and provides instructions for running the code.

- `*.png`  
  - Sample plots (dataset statistics, ROC curves, AUC bar charts) generated during training.

---

## Requirements

- Python 3.7+  
- PyTorch  
- PyTorch Geometric  
- scikit-learn  
- tqdm  
- matplotlib  

Install via:

```bash
pip install torch torchvision torchaudio torch-geometric scikit-learn tqdm matplotlib
