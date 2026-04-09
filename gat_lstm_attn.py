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
    def __init__(self, in_dim=5, hidden_dim=64, out_dim=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


# =========================
# LSTM
# =========================
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out   # [batch, T, H]


# =========================
# MULTI-HEAD SELF-ATTENTION
# =========================
class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Node-conditioned query from each node's final hidden state.
        query = self.query_proj(x[:, -1:, :])
        attn_out, _ = self.attn(query, x, x, need_weights=False)
        return attn_out.squeeze(1)


# =========================
# CLASSIFIER
# =========================
class NodeClassifier(nn.Module):
    def __init__(self, in_dim=64, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# =========================
# FULL MODEL
# =========================
class ATGCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GATEncoder(in_dim=5, hidden_dim=64, out_dim=64, heads=4)
        self.lstm = TemporalLSTM(input_dim=64, hidden_dim=64)
        self.lstm_norm = nn.LayerNorm(64)
        self.attn = TemporalSelfAttention(hidden_dim=64, num_heads=4, dropout=0.1)
        self.attn_norm = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.1)
        self.classifier = NodeClassifier(in_dim=64, num_classes=3)

    def forward(self, x_batch, edge_index, num_nodes, edge_index_cache):
        """Forward pass for the model. Properly structured for PyTorch."""
        B, T, N, F = x_batch.shape
        x_graph = x_batch.reshape(B * T, N, F)
        x_flat = x_graph.reshape(B * T * N, F).to(edge_index.device)

        edge_bt = get_batched_edge_index(
            edge_index, num_nodes, B * T, edge_index.device, edge_index_cache
        )

        gat_out = self.encoder(x_flat, edge_bt)
        G = gat_out.shape[-1]
        gat_out = gat_out.reshape(B, T, N, G)
        gat_out = gat_out.permute(0, 2, 1, 3).reshape(B * N, T, G)

        lstm_out = self.lstm(gat_out)
        lstm_out = self.lstm_norm(lstm_out)

        attn_context = self.attn(lstm_out)
        context = lstm_out[:, -1, :] + attn_context
        context = self.attn_norm(context)
        context = self.dropout(context)

        H = context.shape[-1]
        context = context.reshape(B, N, H)
        return self.classifier(context)


def get_batched_edge_index(base_edge_index, num_nodes, batch_size, device, edge_index_cache):
    if batch_size not in edge_index_cache:
        offsets = torch.arange(batch_size, device=device, dtype=base_edge_index.dtype) * num_nodes
        batched_edge_index = base_edge_index.unsqueeze(-1) + offsets.view(1, 1, -1)
        edge_index_cache[batch_size] = batched_edge_index.reshape(2, -1)
    return edge_index_cache[batch_size]


def forward_logits_batch(model, x_batch, base_edge_index, num_nodes, device, edge_index_cache):
    # x_batch: [B, T, N, F] -> use model's forward() method directly
    x_batch = x_batch.to(device)
    base_edge_index = base_edge_index.to(device)
    return model(x_batch, base_edge_index, num_nodes, edge_index_cache)


# =========================
# TRAIN
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

    # Shuffle training order each epoch.
    num_windows = batch_x.shape[0]
    perm = torch.randperm(num_windows)
    batch_x = batch_x[perm]
    batch_y = batch_y[perm]

    batch_x = batch_x[..., [0,1,2,4,5]]  # remove leakage

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mb_size = x_mb.shape[0]
        total_loss += loss.item() * mb_size
        total_samples += mb_size

    return total_loss / max(total_samples, 1)


# =========================
# EVALUATE
# =========================
def evaluate(model, batch_x, batch_y, edge_index, device, batch_size, num_nodes, edge_index_cache):

    model.eval()

    batch_x = batch_x[..., [0,1,2,4,5]]

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


def predict_logits(model, batch_x, edge_index, device, batch_size, num_nodes, edge_index_cache):
    model.eval()
    batch_x = batch_x[..., [0,1,2,4,5]]

    logits_all = []
    with torch.no_grad():
        for x_mb in torch.split(batch_x, batch_size, dim=0):
            logits = forward_logits_batch(
                model, x_mb, edge_index, num_nodes, device, edge_index_cache
            )
            logits_all.append(logits.cpu())

    return torch.cat(logits_all, dim=0)


def collect_predictions(model, batch_x, batch_y, edge_index, device, batch_size, num_nodes, edge_index_cache):
    logits = predict_logits(
        model,
        batch_x,
        edge_index,
        device,
        batch_size,
        num_nodes,
        edge_index_cache,
    )

    probs = torch.softmax(logits, dim=2)
    preds = logits.argmax(dim=2)

    y_true = batch_y.reshape(-1).cpu().numpy()
    y_pred = preds.reshape(-1).cpu().numpy()
    y_prob = probs.reshape(-1, probs.shape[-1]).cpu().numpy()

    return y_true, y_pred, y_prob


def compute_and_plot_metrics(y_true, y_pred, y_prob, model_name, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    class_names = ["class_0", "class_1", "class_2"]
    num_classes = len(class_names)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
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
    print()
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1 Score  (macro): {f1:.4f}")
    print()
    print(f"Precision (weighted): {precision_w:.4f}")
    print(f"Recall    (weighted): {recall_w:.4f}")
    print(f"F1 Score  (weighted): {f1_w:.4f}")
    print()
    for i in range(num_classes):
        print(f"  Class {i} | Prec: {prec_cls[i]:.4f} | Rec: {rec_cls[i]:.4f} | F1: {f1_cls[i]:.4f}")
    print()
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

    windows, targets, adj, _, _ = load_dataset("./data")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(windows, targets)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index)
    edge_index = edge_index.to(device)

    labels_flat = y_train.view(-1).numpy()
    class_counts = np.bincount(labels_flat, minlength=3)
    total_samples = labels_flat.size
    num_classes = 3
    class_weights = torch.tensor(
        [total_samples / (num_classes * max(count, 1)) for count in class_counts],
        dtype=torch.float32,
    ).to(device)

    print(f"Label distribution: {dict(zip(range(3), class_counts))}")
    print(f"Computed class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = ATGCNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    batch_size = 64
    num_nodes = X_train.shape[2]
    edge_index_cache = {}
    best_val_acc = 0.0
    best_model_path = "gat_lstm_attn_best.pt"

    for epoch in range(70):
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

        scheduler.step(acc)

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

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
    compute_and_plot_metrics(y_true, y_pred, y_prob, model_name="GAT_LSTM_ATTN")