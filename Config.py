"""
config.py — Central configuration for all hyperparameters, paths, and settings.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
MODELS_DIR  = RESULTS_DIR / "models"

for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_PATH   = None          # Set via CLI; falls back to synthetic data
LABEL_COL      = "label"       # Column name for ground-truth labels in CSV
BENIGN_LABEL   = "BenignTraffic"
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
N_SYNTHETIC    = 50_000        # Rows to generate when using synthetic data
SYNTHETIC_ATTACK_RATIO = 0.35  # 35% attack traffic in synthetic data

# Feature columns to DROP (meta / leakage columns common in CIC datasets)
DROP_COLS = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
    "Protocol", "Timestamp", "Flow Duration",
]

# ─── Autoencoder ──────────────────────────────────────────────────────────────
AE_HIDDEN_DIMS  = [256, 128, 64]   # Encoder hidden layer sizes (decoder mirrors)
AE_LATENT_DIM   = 32               # Bottleneck / embedding dimension
AE_DROPOUT      = 0.2
AE_BATCH_SIZE   = 512
AE_EPOCHS       = 50
AE_LR           = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_PATIENCE     = 8                # Early stopping patience
TRAIN_ON_BENIGN_ONLY = True        # Train autoencoder only on benign traffic

# ─── Clustering ───────────────────────────────────────────────────────────────
CLUSTERING_METHOD   = "kmeans"     # "kmeans" | "hdbscan"
N_CLUSTERS_KMEANS   = 10
HDBSCAN_MIN_CLUSTER = 50
HDBSCAN_MIN_SAMPLES = 10

# ─── Jensen-Shannon Divergence ────────────────────────────────────────────────
JSD_N_BINS          = 30           # Histogram bins for empirical PDF estimation
JSD_THRESHOLD       = 0.3          # Clusters with mean JSD > this → anomalous
# Number of latent dimensions used when computing per-cluster JSD
JSD_DIMS_TO_USE     = 8            # Top-8 principal latent dims (avoids noise)

# ─── UMAP (for 2-D visualization only) ───────────────────────────────────────
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.1
UMAP_N_COMPONENTS = 2
UMAP_METRIC       = "euclidean"

# ─── Misc ─────────────────────────────────────────────────────────────────────
DEVICE = "cpu"    # "cuda" if you have a GPU
SEED   = 42
