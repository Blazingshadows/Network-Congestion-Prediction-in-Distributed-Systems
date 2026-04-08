import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

from dataset_generator import load_dataset, split_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# =========================
# GAT ENCODER
# =========================
class GATEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=32, out_dim=32, heads=4):
        super().__init__()

        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


# =========================
# CLASSIFIER
# =========================
class NodeClassifier(nn.Module):
    def __init__(self, in_dim=32, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# =========================
# FULL MODEL
# =========================
class GATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GATEncoder()
        self.classifier = NodeClassifier()

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return self.classifier(x)


def get_batched_edge_index(base_edge_index, num_nodes, batch_size, device, edge_index_cache):
    if batch_size not in edge_index_cache:
        offsets = torch.arange(batch_size, device=device, dtype=base_edge_index.dtype) * num_nodes
        batched_edge_index = base_edge_index.unsqueeze(-1) + offsets.view(1, 1, -1)
        edge_index_cache[batch_size] = batched_edge_index.reshape(2, -1)
    return edge_index_cache[batch_size]


def forward_logits_batch(model, x_batch, base_edge_index, num_nodes, device, edge_index_cache):
    # x_batch: [B, N, F] -> flatten to one disconnected super-graph
    batch_size = x_batch.shape[0]
    x_flat = x_batch.reshape(batch_size * num_nodes, x_batch.shape[-1]).to(device)
    batched_edge_index = get_batched_edge_index(
        base_edge_index, num_nodes, batch_size, device, edge_index_cache
    )

    logits = model(x_flat, batched_edge_index)
    return logits.reshape(batch_size, num_nodes, -1)


# =========================
# TRAIN FUNCTION
# =========================
def train_one_epoch(
    model,
    optimizer,
    criterion,
    batch_x,
    batch_y,
    edge_index,
    device,
    batch_size,
    num_nodes,
    edge_index_cache,
):

    model.train()

    batch_x = batch_x[:, -1, :, :]   # [W, N, F]

    total_loss = 0.0
    total_samples = 0

    x_splits = torch.split(batch_x, batch_size, dim=0)
    y_splits = torch.split(batch_y, batch_size, dim=0)

    for x_mb, y_mb in zip(x_splits, y_splits):
        logits = forward_logits_batch(
            model, x_mb, edge_index, num_nodes, device, edge_index_cache
        )
        y_mb = y_mb.to(device)

        loss = criterion(logits.reshape(-1, 3), y_mb.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mb_size = x_mb.shape[0]
        total_loss += loss.item() * mb_size
        total_samples += mb_size

    return total_loss / max(total_samples, 1)


# =========================
# EVALUATION
# =========================
def evaluate(model, batch_x, batch_y, edge_index, device, batch_size, num_nodes, edge_index_cache):
    model.eval()

    batch_x = batch_x[:, -1, :, :]
    correct = 0
    total = 0

    with torch.no_grad():
        x_splits = torch.split(batch_x, batch_size, dim=0)
        y_splits = torch.split(batch_y, batch_size, dim=0)

        for x_mb, y_mb in zip(x_splits, y_splits):
            logits = forward_logits_batch(
                model, x_mb, edge_index, num_nodes, device, edge_index_cache
            )
            preds = logits.argmax(dim=2)
            y_mb = y_mb.to(device)

            correct += (preds == y_mb).sum().item()
            total += y_mb.numel()

    return correct / total


def collect_predictions(model, batch_x, batch_y, edge_index, device, batch_size, num_nodes, edge_index_cache):
    model.eval()

    batch_x = batch_x[:, -1, :, :]
    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        x_splits = torch.split(batch_x, batch_size, dim=0)
        y_splits = torch.split(batch_y, batch_size, dim=0)

        for x_mb, y_mb in zip(x_splits, y_splits):
            logits = forward_logits_batch(
                model, x_mb, edge_index, num_nodes, device, edge_index_cache
            )
            probs = torch.softmax(logits, dim=2)
            preds = logits.argmax(dim=2)

            all_true.append(y_mb.reshape(-1).cpu().numpy())
            all_pred.append(preds.reshape(-1).cpu().numpy())
            all_prob.append(probs.reshape(-1, probs.shape[-1]).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    return y_true, y_pred, y_prob


def compute_and_plot_metrics(y_true, y_pred, y_prob, model_name, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    class_names = ["class_0", "class_1", "class_2"]
    num_classes = len(class_names)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    try:
        auc_macro_ovr = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc_macro_ovr = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(7, 5))
    for class_idx in range(num_classes):
        positives = y_true_bin[:, class_idx].sum()
        negatives = (1 - y_true_bin[:, class_idx]).sum()

        if positives == 0 or negatives == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
        class_auc = roc_auc_score(y_true_bin[:, class_idx], y_prob[:, class_idx])
        plt.plot(fpr, tpr, label=f"{class_names[class_idx]} (AUC={class_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"{model_name.lower()}_roc.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()

    print(f"\n{model_name} Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if np.isnan(auc_macro_ovr):
        print("AUC/ROC  : N/A (insufficient class diversity in targets)")
    else:
        print(f"AUC/ROC  : {auc_macro_ovr:.4f} (macro OVR)")
    print(f"Confusion matrix figure saved to: {cm_path}")
    print(f"ROC figure saved to: {roc_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # LOAD DATA
    # =========================
    windows, targets, adj, _, _ = load_dataset("./data")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(windows, targets)

    # Convert to torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # =========================
    # REMOVE FEATURE LEAKAGE (drop bandwidth_util index=3)
    # =========================
    X_train = X_train[..., [0,1,2,4,5]]
    X_val   = X_val[..., [0,1,2,4,5]]
    X_test  = X_test[..., [0,1,2,4,5]]

    # =========================
    # CLASS WEIGHTS
    # =========================
    labels_flat = y_train.view(-1).numpy()
    class_counts = np.bincount(labels_flat)

    class_weights = torch.tensor([0.3, 0.4, 0.6], dtype=torch.float32).to(device)
    

    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # =========================
    # EDGE INDEX
    # =========================
    edge_index = np.array(np.nonzero(adj))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_index, _ = add_self_loops(edge_index)
    edge_index = edge_index.to(device)

    # =========================
    # MODEL
    # =========================
    model = GATModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    batch_size = 64
    num_nodes = X_train.shape[2]
    edge_index_cache = {}

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(50):
        loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            X_train,
            y_train,
            edge_index,
            device,
            batch_size,
            num_nodes,
            edge_index_cache,
        )
        acc = evaluate(
            model,
            X_val,
            y_val,
            edge_index,
            device,
            batch_size,
            num_nodes,
            edge_index_cache,
        )

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    # =========================
    # COMPARABLE METRICS + FIGURES (TEST SPLIT)
    # =========================
    y_true, y_pred, y_prob = collect_predictions(
        model,
        X_test,
        y_test,
        edge_index,
        device,
        batch_size,
        num_nodes,
        edge_index_cache,
    )
    compute_and_plot_metrics(y_true, y_pred, y_prob, model_name="GAT")