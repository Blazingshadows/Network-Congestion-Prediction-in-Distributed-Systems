"""
AT-GCN Dataset Generator
========================
Generates synthetic spatiotemporal network congestion data using:
  - Abilene 12-node topology (hardcoded)
  - AR(1) correlated traffic model for temporal realism
  - M/M/1 queuing formulas for feature computation
  - Spatial congestion propagation to neighbors
  - Threshold-based label generation (Normal / Warning / Congested)

Output shapes:
  windows      : [num_windows, W, N, F]  = [num_windows, 20, 12, 6]
  labels       : [num_windows, N]         = [num_windows, 12]
  adj_matrix   : [N, N]                   = [12, 12]
"""

import numpy as np
import os


# ---------------------------------------------------------------------------
# 1. CONSTANTS & CONFIGURATION
# ---------------------------------------------------------------------------

# Number of nodes in Abilene topology
N_NODES = 12

# Feature names (order matters — must stay consistent with model)
# [RTT, queue_depth, packet_loss, bandwidth_util, jitter, throughput]
N_FEATURES = 6

# Sliding window size (timesteps per sample)
WINDOW_SIZE = 20

# Label thresholds (based on bandwidth utilization, feature index 3)
LABEL_NORMAL    = 0   # util < 0.50
LABEL_WARNING   = 1   # 0.50 <= util < 0.80
LABEL_CONGESTED = 2   # util >= 0.80

THRESH_WARNING   = 0.50
THRESH_CONGESTED = 0.80

# Link capacity in Mbps (Abilene links are ~10 Gbps; we use scaled version)
LINK_CAPACITY_MBPS = 1000.0

# Service rate mu for M/M/1 queuing (packets per ms)
MU = 1.0

# AR(1) process parameters
AR_PHI   = 0.85   # autocorrelation coefficient (how much past affects present)
AR_SIGMA = 0.08   # noise standard deviation

# Spatial propagation coefficient (how much congestion bleeds to neighbors)
SPATIAL_ALPHA = 0.25

# Base load range [min, max] as fraction of link capacity
BASE_LOAD_MIN = 0.1
BASE_LOAD_MAX = 0.6


# ---------------------------------------------------------------------------
# 2. ABILENE TOPOLOGY
# ---------------------------------------------------------------------------

def build_abilene_topology():
    """
    Returns the adjacency matrix for the 12-node Abilene backbone topology.

    Nodes (indices):
      0: Seattle       1: Sunnyvale     2: Los Angeles
      3: Denver        4: Kansas City   5: Houston
      6: Indianapolis  7: Atlanta       8: Washington DC
      9: New York     10: Chicago      11: Pittsburgh

    Returns
    -------
    adj : np.ndarray, shape [12, 12], dtype float32
        Symmetric binary adjacency matrix (1 = link exists, 0 = no link).
    """
    edges = [
        (0, 1),   # Seattle    -- Sunnyvale
        (0, 3),   # Seattle    -- Denver
        (1, 2),   # Sunnyvale  -- Los Angeles
        (1, 3),   # Sunnyvale  -- Denver
        (2, 5),   # LA         -- Houston
        (3, 4),   # Denver     -- Kansas City
        (4, 5),   # Kansas City-- Houston
        (4, 6),   # Kansas City-- Indianapolis
        (5, 7),   # Houston    -- Atlanta
        (6, 7),   # Indianapolis--Atlanta
        (6, 10),  # Indianapolis--Chicago
        (7, 8),   # Atlanta    -- Washington DC
        (8, 9),   # Washington -- New York
        (8, 11),  # Washington -- Pittsburgh
        (9, 10),  # New York   -- Chicago
        (10, 11), # Chicago    -- Pittsburgh
    ]

    adj = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0   # undirected

    return adj


# ---------------------------------------------------------------------------
# 3. AR(1) TRAFFIC SIMULATION
# ---------------------------------------------------------------------------

def simulate_ar1_traffic(T, n_nodes, phi=AR_PHI, sigma=AR_SIGMA,
                          base_min=BASE_LOAD_MIN, base_max=BASE_LOAD_MAX,
                          seed=None):
    """
    Simulate correlated traffic load for each node over T timesteps
    using an AR(1) process.

    AR(1):  x_t = phi * x_{t-1} + epsilon_t,  epsilon ~ N(0, sigma^2)

    The process is shifted to a per-node base load so nodes have
    different average utilization levels (heterogeneous network).

    Parameters
    ----------
    T        : int   — number of timesteps
    n_nodes  : int   — number of nodes
    phi      : float — AR(1) autocorrelation coefficient
    sigma    : float — noise std dev
    base_min : float — minimum base load fraction
    base_max : float — maximum base load fraction
    seed     : int   — random seed for reproducibility

    Returns
    -------
    load : np.ndarray, shape [T, n_nodes]
        Load values in [0, 1] representing fraction of link capacity used.
    """
    rng = np.random.RandomState(seed)

    # Each node gets a different base load (heterogeneous traffic)
    base_loads = rng.uniform(base_min, base_max, size=n_nodes)

    load = np.zeros((T, n_nodes), dtype=np.float32)
    load[0] = base_loads  # initialise at base

    for t in range(1, T):
        noise = rng.normal(0, sigma, size=n_nodes)
        # AR(1) around base load (mean-reverting)
        load[t] = base_loads + phi * (load[t-1] - base_loads) + noise

    # Clip to valid range [0.01, 0.99] — avoid division by zero in M/M/1
    load = np.clip(load, 0.01, 0.99)

    return load


# ---------------------------------------------------------------------------
# 4. SPATIAL CONGESTION PROPAGATION
# ---------------------------------------------------------------------------

def propagate_congestion(load, adj, alpha=SPATIAL_ALPHA):
    """
    Spread congestion from congested nodes to their neighbors.

    For each timestep, a node's effective load is:
        effective_load[t, i] = load[t, i]
                             + alpha * mean(load[t, neighbors_of_i])

    This models the real-world effect where a congested router increases
    queuing at adjacent routers.

    Parameters
    ----------
    load  : np.ndarray, shape [T, N]
    adj   : np.ndarray, shape [N, N]
    alpha : float — propagation coefficient

    Returns
    -------
    effective_load : np.ndarray, shape [T, N], clipped to [0.01, 0.99]
    """
    T, N = load.shape
    effective_load = load.copy()

    # Degree of each node (number of neighbors)
    degree = adj.sum(axis=1, keepdims=True)  # [N, 1]
    degree = np.where(degree == 0, 1, degree)  # avoid divide-by-zero

    for t in range(T):
        # Neighbor average load for each node: [N]
        neighbor_load_sum = adj @ load[t]          # [N]
        neighbor_load_avg = neighbor_load_sum / degree.squeeze()
        effective_load[t] = load[t] + alpha * neighbor_load_avg

    effective_load = np.clip(effective_load, 0.01, 0.99)
    return effective_load


# ---------------------------------------------------------------------------
# 5. FEATURE COMPUTATION (M/M/1 QUEUING)
# ---------------------------------------------------------------------------

def compute_features(effective_load, capacity_mbps=LINK_CAPACITY_MBPS,
                     mu=MU):
    """
    Compute 6 network features per node per timestep using M/M/1 queuing
    theory, where rho = effective_load (traffic intensity = lambda/mu).

    M/M/1 formulas used:
      - rho       = lambda / mu  (traffic intensity, = effective_load here)
      - L_q       = rho^2 / (1 - rho)             [mean queue length]
      - W_q       = rho / (mu * (1 - rho))         [mean queuing delay]
      - throughput= rho * capacity                  [actual throughput]
      - loss prob ≈ rho^k for large k (approx heavy-traffic loss)

    Features (in order, matching N_FEATURES=6):
      0: RTT (ms)               — base_rtt + queuing delay
      1: Queue depth            — mean number of packets in queue
      2: Packet loss rate       — approximated from utilization
      3: Bandwidth utilization  — raw effective_load (0 to 1)
      4: Jitter (ms)            — variance-based delay variation
      5: Throughput (Mbps)      — actual carried traffic

    Parameters
    ----------
    effective_load : np.ndarray, shape [T, N]
    capacity_mbps  : float — link capacity in Mbps
    mu             : float — service rate

    Returns
    -------
    features : np.ndarray, shape [T, N, 6]  — NOT yet normalised
    """
    T, N = effective_load.shape
    rho = effective_load  # shape [T, N], values in (0, 1)

    # --- M/M/1 derived quantities ---
    # Avoid numerical issues near rho=1
    safe_denom = np.maximum(1 - rho, 1e-4)

    # Queue length: E[L_q] = rho^2 / (1 - rho)
    queue_depth = (rho ** 2) / safe_denom                   # [T, N]

    # Queuing delay: W_q = rho / (mu * (1 - rho))
    queuing_delay = rho / (mu * safe_denom)                 # [T, N]

    # RTT = base propagation delay (10ms) + queuing delay (scaled)
    base_rtt = 10.0  # ms
    rtt = base_rtt + queuing_delay * 5.0                    # [T, N]

    # Packet loss — approximation: loss rises steeply above 80% util
    # Using: loss ~ rho^10 (near-zero below threshold, steep above)
    packet_loss = np.clip(rho ** 10, 0, 1)                  # [T, N]

    # Bandwidth utilization — direct
    bandwidth_util = rho                                     # [T, N]

    # Jitter — approximation: std dev of queuing delay
    # In M/M/1: Var[W_q] = rho / (mu^2 * (1-rho)^2)
    jitter = np.sqrt(np.maximum(rho / (mu**2 * safe_denom**2), 0)) * 2.0
    jitter = np.clip(jitter, 0, 50)                         # [T, N]

    # Throughput in Mbps
    throughput = rho * capacity_mbps                        # [T, N]

    # Stack into [T, N, 6]
    features = np.stack([
        rtt,              # feature 0
        queue_depth,      # feature 1
        packet_loss,      # feature 2
        bandwidth_util,   # feature 3
        jitter,           # feature 4
        throughput,       # feature 5
    ], axis=-1).astype(np.float32)

    return features


# ---------------------------------------------------------------------------
# 6. FEATURE NORMALISATION
# ---------------------------------------------------------------------------

def normalize_features(features):
    """
    Min-max normalise each feature independently across all timesteps
    and nodes, scaling each to [0, 1].

    Parameters
    ----------
    features : np.ndarray, shape [T, N, F]

    Returns
    -------
    norm_features : np.ndarray, shape [T, N, F]
    feat_min      : np.ndarray, shape [F,]  — for inverse transform
    feat_max      : np.ndarray, shape [F,]  — for inverse transform
    """
    T, N, F = features.shape
    flat = features.reshape(-1, F)  # [T*N, F]

    feat_min = flat.min(axis=0)     # [F]
    feat_max = flat.max(axis=0)     # [F]

    denom = np.where((feat_max - feat_min) == 0, 1, feat_max - feat_min)
    norm_flat = (flat - feat_min) / denom

    norm_features = norm_flat.reshape(T, N, F).astype(np.float32)
    return norm_features, feat_min, feat_max


# ---------------------------------------------------------------------------
# 7. LABEL GENERATION
# ---------------------------------------------------------------------------

def generate_labels(effective_load):
    """
    Assign congestion labels to each node at each timestep based on
    bandwidth utilization (= effective_load).

    Thresholds:
      Normal    (0): util < 0.50
      Warning   (1): 0.50 <= util < 0.80
      Congested (2): util >= 0.80

    Parameters
    ----------
    effective_load : np.ndarray, shape [T, N]

    Returns
    -------
    labels : np.ndarray, shape [T, N], dtype int64
    """
    labels = np.zeros_like(effective_load, dtype=np.int64)
    labels[effective_load >= THRESH_WARNING]   = LABEL_WARNING
    labels[effective_load >= THRESH_CONGESTED] = LABEL_CONGESTED
    return labels


# ---------------------------------------------------------------------------
# 8. SLIDING WINDOW CREATION
# ---------------------------------------------------------------------------

def create_windows(features, labels, window_size=WINDOW_SIZE):
    """
    Slice the full time series into overlapping windows.

    For a window ending at timestep t:
      X[t] = features[t - W : t]        shape [W, N, F]
      y[t] = labels[t]                   shape [N]

    We predict the label at the LAST timestep of each window.

    Parameters
    ----------
    features    : np.ndarray, shape [T, N, F]
    labels      : np.ndarray, shape [T, N]
    window_size : int

    Returns
    -------
    windows : np.ndarray, shape [num_windows, W, N, F]
    targets : np.ndarray, shape [num_windows, N]
    """
    T = features.shape[0]
    num_windows = T - window_size

    windows = np.zeros((num_windows, window_size,
                        N_NODES, N_FEATURES), dtype=np.float32)
    targets = np.zeros((num_windows, N_NODES), dtype=np.int64)

    for i in range(num_windows):
        windows[i] = features[i : i + window_size]   # [W, N, F]
        targets[i] = labels[i + window_size]          # [N]  — label at end of window

    return windows, targets


# ---------------------------------------------------------------------------
# 9. MAIN GENERATION FUNCTION
# ---------------------------------------------------------------------------

def generate_dataset(T=1000, window_size=WINDOW_SIZE, seed=42, verbose=True):
    """
    Full pipeline: topology → traffic → propagation → features →
                   normalisation → labels → windows.

    Parameters
    ----------
    T           : int  — total number of timesteps to simulate
    window_size : int  — sliding window size
    seed        : int  — random seed
    verbose     : bool — print shape info

    Returns
    -------
    windows    : np.ndarray, shape [num_windows, W, N, F]
    targets    : np.ndarray, shape [num_windows, N]
    adj        : np.ndarray, shape [N, N]
    feat_min   : np.ndarray, shape [F,]
    feat_max   : np.ndarray, shape [F,]
    """

    if verbose:
        print("=" * 55)
        print("AT-GCN Dataset Generator")
        print("=" * 55)

    # Step 1: Topology
    adj = build_abilene_topology()
    if verbose:
        print(f"[1] Topology       : {N_NODES} nodes, "
              f"{int(adj.sum() / 2)} edges")

    # Step 2: AR(1) traffic simulation
    load = simulate_ar1_traffic(T, N_NODES, seed=seed)
    if verbose:
        print(f"[2] AR(1) traffic  : shape {load.shape}, "
              f"mean={load.mean():.3f}, std={load.std():.3f}")

    # Step 3: Spatial propagation
    effective_load = propagate_congestion(load, adj)
    if verbose:
        print(f"[3] After propagat.: shape {effective_load.shape}, "
              f"mean={effective_load.mean():.3f}")

    # Step 4: Feature computation
    features = compute_features(effective_load)
    if verbose:
        print(f"[4] Raw features   : shape {features.shape}")

    # Step 5: Normalisation
    norm_features, feat_min, feat_max = normalize_features(features)
    if verbose:
        print(f"[5] Norm features  : shape {norm_features.shape}, "
              f"range [{norm_features.min():.2f}, {norm_features.max():.2f}]")

    # Step 6: Labels
    labels = generate_labels(effective_load)
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "Normal", 1: "Warning", 2: "Congested"}
    if verbose:
        print(f"[6] Labels         : shape {labels.shape}")
        for u, c in zip(unique, counts):
            pct = 100 * c / labels.size
            print(f"      {label_names[u]:>10} ({u}): "
                  f"{c:6d} samples ({pct:.1f}%)")

    # Step 7: Sliding windows
    windows, targets = create_windows(norm_features, labels, window_size)
    if verbose:
        print(f"[7] Windows        : {windows.shape}  "
              f"(num_windows, W, N, F)")
        print(f"    Targets        : {targets.shape}  "
              f"(num_windows, N)")
        print("=" * 55)

    return windows, targets, adj, feat_min, feat_max


# ---------------------------------------------------------------------------
# 10. TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------------

def split_dataset(windows, targets, train_ratio=0.7, val_ratio=0.15,
                  verbose=True):
    """
    Chronological split — no shuffling, to preserve temporal ordering.

    Ratios: 70% train / 15% val / 15% test

    Parameters
    ----------
    windows     : np.ndarray, shape [num_windows, W, N, F]
    targets     : np.ndarray, shape [num_windows, N]
    train_ratio : float
    val_ratio   : float

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    n = len(windows)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    X_train, y_train = windows[:train_end],       targets[:train_end]
    X_val,   y_val   = windows[train_end:val_end], targets[train_end:val_end]
    X_test,  y_test  = windows[val_end:],          targets[val_end:]

    if verbose:
        print(f"Train : {X_train.shape[0]} windows")
        print(f"Val   : {X_val.shape[0]} windows")
        print(f"Test  : {X_test.shape[0]} windows")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------------------------------------------------------------------------
# 11. SAVE / LOAD UTILITIES
# ---------------------------------------------------------------------------

def save_dataset(save_dir, windows, targets, adj, feat_min, feat_max):
    """Save all arrays to disk as .npy files."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "windows.npy"),   windows)
    np.save(os.path.join(save_dir, "targets.npy"),   targets)
    np.save(os.path.join(save_dir, "adj.npy"),       adj)
    np.save(os.path.join(save_dir, "feat_min.npy"),  feat_min)
    np.save(os.path.join(save_dir, "feat_max.npy"),  feat_max)
    print(f"Dataset saved to: {save_dir}")


def load_dataset(save_dir):
    """Load all arrays from disk."""
    windows  = np.load(os.path.join(save_dir, "windows.npy"))
    targets  = np.load(os.path.join(save_dir, "targets.npy"))
    adj      = np.load(os.path.join(save_dir, "adj.npy"))
    feat_min = np.load(os.path.join(save_dir, "feat_min.npy"))
    feat_max = np.load(os.path.join(save_dir, "feat_max.npy"))
    print(f"Dataset loaded from: {save_dir}")
    return windows, targets, adj, feat_min, feat_max


# ---------------------------------------------------------------------------
# 12. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Generate
    windows, targets, adj, feat_min, feat_max = generate_dataset(
        T=2000,
        window_size=WINDOW_SIZE,
        seed=42,
        verbose=True
    )

    # Split
    print("\nDataset split:")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
        windows, targets, verbose=True
    )

    # Verify shapes
    print("\nFinal shape verification:")
    print(f"  X_train : {X_train.shape}   expected (?, {WINDOW_SIZE}, {N_NODES}, {N_FEATURES})")
    print(f"  y_train : {y_train.shape}   expected (?, {N_NODES})")
    print(f"  adj     : {adj.shape}       expected ({N_NODES}, {N_NODES})")

    # Save
    save_dataset("./data", windows, targets, adj, feat_min, feat_max)