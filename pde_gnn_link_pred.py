import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    train_test_split_edges,
    negative_sampling,
    add_self_loops,
    degree,
)
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------
# 0. Fix random seed for reproducibility
# ---------------------------------------------
torch.manual_seed(42)

# ---------------------------------------------
# 1. Load PubMed and split edges
# ---------------------------------------------
print("\n=== Loading PubMed dataset for link prediction ===")
dataset = Planetoid(root='data/PubMed', name='PubMed', transform=NormalizeFeatures())
data = dataset[0]

print("Splitting edges into train/validation/test sets...")
data = train_test_split_edges(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ---------------------------------------------
# 2. Print & plot dataset stats
# ---------------------------------------------
num_nodes = data.num_nodes
num_train_pos = data.train_pos_edge_index.size(1)
num_val_pos = data.val_pos_edge_index.size(1)
num_test_pos = data.test_pos_edge_index.size(1)

print(f"\nPubMed dataset statistics (link-prediction mode):")
print(f"  • Number of nodes:       {num_nodes}")
print(f"  • Number of train edges: {num_train_pos}")
print(f"  • Number of val edges:   {num_val_pos}")
print(f"  • Number of test edges:  {num_test_pos}")

plt.figure(figsize=(6, 4))
plt.bar(
    ['Nodes', 'Train edges', 'Val edges', 'Test edges'],
    [num_nodes, num_train_pos, num_val_pos, num_test_pos],
    color=['skyblue', 'lightgreen', 'lightcoral', 'plum']
)
plt.title("PubMed Link-Prediction Dataset Statistics")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("pubmed_dataset_stats.png")
print("Saved plot: pubmed_dataset_stats.png")
plt.show()

# ---------------------------------------------
# 3. Baseline GCN for link prediction
# ---------------------------------------------
class GCNLinkPredict(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNLinkPredict, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        return pos_scores, neg_scores

# ---------------------------------------------
# 4. Optimized PDE-GNN layer (no Python loops)
# ---------------------------------------------
class PDEGNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.05, beta=1.5):
        super(PDEGNNLayer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 1) Linear projection
        h = self.lin(x)  # shape: [num_nodes, out_channels]

        # 2) Build normalized adjacency for diffusion
        edge_index_norm, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        row, col = edge_index_norm

        deg = degree(row, num_nodes=h.size(0), dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 3) Create sparse adjacency matrix (COO) and multiply by h
        adj_t = torch.sparse_coo_tensor(
            edge_index_norm,
            norm,
            (h.size(0), h.size(0))
        ).coalesce()

        # sparse->dense matmul: shape [num_nodes, out_channels]
        diffusion_part = torch.sparse.mm(adj_t, h)

        # 4) Diffusion term = alpha * (Â h - h)
        diffusion = self.alpha * (diffusion_part - h)

        # 5) Reaction term = beta * ReLU(h)
        reaction = self.beta * torch.relu(h)

        # 6) Update
        return h + diffusion + reaction

class PDEGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, alpha=0.05, beta=1.5):
        super(PDEGNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        # First layer: input -> hidden
        self.layers.append(PDEGNNLayer(in_channels, hidden_channels, alpha, beta))
        # (No middle layers for num_layers=2)
        # Last layer: hidden -> hidden
        if num_layers > 1:
            self.layers.append(PDEGNNLayer(hidden_channels, hidden_channels, alpha, beta))

    def encode(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        return pos_scores, neg_scores

# Utility to create edge labels
def get_edge_labels(pos_edge_index, neg_edge_index):
    pos_labels = torch.ones(pos_edge_index.size(1), device=device)
    neg_labels = torch.zeros(neg_edge_index.size(1), device=device)
    return torch.cat([pos_labels, neg_labels])

# ---------------------------------------------
# 5. Training & evaluation (with tqdm)
# ---------------------------------------------
def train_and_evaluate(model, epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in tqdm(range(1, epochs + 1), desc=f"Training {model.__class__.__name__}"):
        model.train()
        optimizer.zero_grad()

        # Sample negative edges
        neg_train_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1),
        )

        pos_train_scores, neg_train_scores = model(
            data.x,
            data.train_pos_edge_index,
            data.train_pos_edge_index,
            neg_train_edge_index
        )
        train_scores = torch.cat([pos_train_scores, neg_train_scores], dim=0)
        train_labels = get_edge_labels(data.train_pos_edge_index, neg_train_edge_index)

        loss = F.binary_cross_entropy(train_scores, train_labels)
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        neg_test_edge_index = negative_sampling(
            edge_index=data.test_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.test_pos_edge_index.size(1),
        )
        pos_test_scores, neg_test_scores = model(
            data.x,
            data.train_pos_edge_index,
            data.test_pos_edge_index,
            neg_test_edge_index
        )
        test_scores = torch.cat([pos_test_scores, neg_test_scores]).cpu().numpy()
        test_labels = get_edge_labels(data.test_pos_edge_index, neg_test_edge_index).cpu().numpy()

        auc = roc_auc_score(test_labels, test_scores)
        fpr, tpr, _ = roc_curve(test_labels, test_scores)

    return auc, fpr, tpr

# ---------------------------------------------
# 6. Train & evaluate baseline GCN
# ---------------------------------------------
print("\n=== Training baseline GCN model on PubMed ===")
gcn_model = GCNLinkPredict(dataset.num_node_features, hidden_channels=64)
gcn_auc, gcn_fpr, gcn_tpr = train_and_evaluate(gcn_model, epochs=100)
print(f"Baseline GCN Test AUC (PubMed): {gcn_auc:.4f}\n")

# ---------------------------------------------
# 7. Train & evaluate PDE-GNN (tuned)
# ---------------------------------------------
print("=== Training PDE-GNN model on PubMed (optimized) ===")
pde_model = PDEGNN(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    num_layers=3,   # fewer layers to speed up
    alpha=0.05,     # smaller diffusion
    beta=1.5        # stronger reaction
)
pde_auc, pde_fpr, pde_tpr = train_and_evaluate(pde_model, epochs=100)
print(f"PDE-GNN Test AUC (PubMed): {pde_auc:.4f}\n")

# ---------------------------------------------
# 8. Plot ROC comparison
# ---------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(
    gcn_fpr,
    gcn_tpr,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'GCN (AUC = {gcn_auc:.3f})'
)
plt.plot(
    pde_fpr,
    pde_tpr,
    color='blue',
    linewidth=2,
    label=f'PDE-GNN (AUC = {pde_auc:.3f})'
)
plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: GCN vs PDE-GNN on PubMed Test Set')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pubmed_roc_comparison.png")
print("Saved plot: pubmed_roc_comparison.png")
plt.show()

# ---------------------------------------------
# 9. Plot AUC bar comparison
# ---------------------------------------------
plt.figure(figsize=(5, 4))
plt.bar(
    ['GCN', 'PDE-GNN'],
    [gcn_auc, pde_auc],
    color=['tomato', 'steelblue']
)
plt.ylabel('Test AUC')
plt.title('Test AUC Comparison on PubMed')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("pubmed_auc_bar_comparison.png")
print("Saved plot: pubmed_auc_bar_comparison.png")
plt.show()

print("\n=== Script finished successfully ===\n")
