# Network Congestion Prediction in Distributed Systems

## GAT+LSTM Attention Optimization Walkthrough

## Project Summary
This project predicts network congestion classes on graph-structured time-series data using three model variants:

- Vanilla GAT (spatial baseline)
- GAT + LSTM (spatiotemporal baseline)
- GAT + LSTM + Attention (optimized best variant)

The workflow focused on identifying and removing training and architecture bottlenecks that were suppressing temporal model performance.

## Changes Made
This session revamped both the data-processing/training pipeline and the attention architecture.

### 1) Vectorized Loop Elimination
- Replaced slow per-epoch `torch_geometric.DataLoader` graph loops.
- Introduced precomputed batched edge-index offsets for replicated static graphs.
- Enabled fast block-style graph batching for thousands of graphs per epoch.

### 2) Mini-Batch Optimization
- Removed single-step full-batch gradient descent behavior.
- Added true mini-batch training with `torch.split` and `batch_size=64`.
- Increased effective gradient updates per epoch, improving convergence for LSTM-based models.

### 3) Residual Self-Attention Tuning
- Upgraded attention to multi-head self-attention with a learnable global query token.
- Added residual fusion (`last_lstm_state + attn_context`) to preserve stable gradient flow.
- Added LayerNorm around sequence and attention outputs for better optimization stability.

### 4) Unified Evaluation and Comparability
- Added consistent metrics and plots across model scripts:
  - Accuracy, Precision, Recall, F1, AUC/ROC
  - Confusion matrix figure
  - ROC curve figure
- Standardized evaluation on identical test split for fair comparison.

## Validation Results
All metrics below are from the same split protocol and evaluation structure.

### Baseline Comparisons (Final)

| Model | Accuracy | F1 Score | AUC/ROC | Status |
|---|---:|---:|---:|---|
| Vanilla GAT | 0.7539 | 0.6435 | 0.8815 | Original spatial baseline |
| GAT + LSTM | 0.7621 | 0.6527 | 0.8847 | Fixed mini-batching (exceeded base) |
| GAT + LSTM + ATTN | 0.7713 | 0.6696 | 0.8973 | Best setup (resolved bottleneck) |

## Milestones

### Milestone 1: Comparable Evaluation Foundation
- Added confusion matrix and ROC outputs for each model.
- Added unified reporting of Accuracy, Precision, Recall, F1, and AUC.

### Milestone 2: Training Dynamics Repair
- Refactored GAT and GAT+LSTM to true mini-batch training.
- Replaced graph-loop bottlenecks with cached batched edge-index strategy.
- Restored stable convergence in recurrent models.

### Milestone 3: Attention Architecture Upgrade
- Baseline-aligned hyperparameters (hidden size and learning rate tuning).
- Multi-head attention with global learnable query.
- Residual connection + LayerNorm stabilization.
- Achieved best end-to-end performance among tested models.

## Model Template (Use This for Any New Variant)
Use this structure when adding future models so comparisons remain clean and reproducible.

### Model Name
- **Architecture**: (e.g., GAT + GRU + Attention)
- **Key Changes**:
  - Change 1
  - Change 2
  - Change 3
- **Training Setup**:
  - batch size: 64
  - optimizer/lr: ...
  - epochs: ...
- **Evaluation Outputs**:
  - Accuracy, Precision, Recall, F1, AUC/ROC
  - Confusion matrix PNG
  - ROC curve PNG
- **Final Metrics**:
  - Accuracy: ...
  - F1: ...
  - AUC/ROC: ...
- **Milestone Outcome**: (baseline / improved / best)

## Repository Files
- `dataset_generator.py`: synthetic spatiotemporal graph dataset generation and splitting
- `gat.py`: vanilla GAT model and evaluation pipeline
- `gat_lstm.py`: GAT + LSTM model with mini-batch graph batching
- `gat_lstm_attn.py`: optimized GAT + LSTM + attention model
- `data/`: serialized dataset artifacts (`windows.npy`, `targets.npy`, `adj.npy`, etc.)
- `figures/`: generated confusion matrix and ROC plots

## Reproducibility Notes
- Keep dataset split strategy unchanged when comparing models.
- Compare models using the same batch size and evaluation logic.
- For robust reporting, run multiple seeds and report mean +- std.
