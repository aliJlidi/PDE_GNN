# How This Implements “PDE → GNN”

## Diffusion Term

\[
\text{diffusion} \;=\; \alpha \bigl(\tilde{A}x - x\bigr)
\]

- We build \(\tilde{A}x\) via normalized adjacency multiplication.
- \(\alpha\) is the diffusion coefficient.

---

## Reaction Term

\[
\text{reaction} \;=\; \beta\,\sigma(Wx)
\]

- A learnable linear transform followed by ReLU acts as the reaction (feature update) term.
- \(\beta\) is the reaction coefficient.

---

## Overall Update

\[
x^{(l+1)} \;=\; x^{(l)} \;+\; \alpha \bigl(\tilde{A}\,x^{(l)} - x^{(l)}\bigr) \;+\; \beta\,\sigma\bigl(W^{(l)}\,x^{(l)}\bigr).
\]

This matches the LaTeX formula:

\[
H^{(l+1)} \;=\; H^{(l)} \;+\; \alpha \bigl(\tilde{A}\,H^{(l)} - H^{(l)}\bigr) \;+\; \beta\,\sigma\bigl(H^{(l)}\,W^{(l)}\bigr).
\]

---

## To Run

1. **Install dependencies** (ideally in a fresh virtual environment):

 
   pip install torch torchvision torchaudio torch-geometric scikit-learn
   python pde_gnn_link_pred.py
