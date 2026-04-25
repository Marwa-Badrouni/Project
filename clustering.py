"""
utils/clustering.py — K-Means and HDBSCAN clustering wrappers.
"""

import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLUSTERING_METHOD, N_CLUSTERS_KMEANS,
    HDBSCAN_MIN_CLUSTER, HDBSCAN_MIN_SAMPLES, RANDOM_STATE
)


def cluster_embeddings(embeddings: np.ndarray,
                        method: str = None) -> np.ndarray:
    """
    Cluster latent embeddings and return integer cluster labels.

    Parameters
    ----------
    embeddings : (N, latent_dim)
    method     : "kmeans" | "hdbscan" (defaults to config value)

    Returns
    -------
    labels : (N,) int array. HDBSCAN may return -1 for noise.
    model  : fitted clustering model
    """
    method = method or CLUSTERING_METHOD

    if method == "kmeans":
        model = KMeans(
            n_clusters=N_CLUSTERS_KMEANS,
            random_state=RANDOM_STATE,
            n_init=10,
            max_iter=300,
        )
        labels = model.fit_predict(embeddings)
        print(f"[Clustering] K-Means k={N_CLUSTERS_KMEANS}  "
              f"inertia={model.inertia_:.2f}")

    elif method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            raise ImportError("Install hdbscan: pip install hdbscan")
        model = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = (labels == -1).sum()
        print(f"[Clustering] HDBSCAN  clusters={n_clusters}  noise={n_noise}")

    else:
        raise ValueError(f"Unknown clustering method: {method!r}")

    _print_cluster_summary(labels)
    return labels, model


def _print_cluster_summary(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print("[Clustering] Cluster sizes:")
    for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        tag = " (noise)" if cid == -1 else ""
        print(f"  Cluster {cid:3d}{tag}: {cnt:6d} samples")


def cluster_purity(cluster_labels: np.ndarray,
                   true_labels: np.ndarray) -> dict:
    """
    For each cluster, compute the fraction of the majority class.
    Also returns attack fraction per cluster.
    """
    unique_ids = np.unique(cluster_labels)
    unique_ids = unique_ids[unique_ids != -1]
    results = {}
    for cid in unique_ids:
        mask      = cluster_labels == cid
        y_cluster = true_labels[mask]
        n         = len(y_cluster)
        n_attack  = int((y_cluster == 1).sum())
        n_benign  = n - n_attack
        purity    = max(n_attack, n_benign) / n
        results[int(cid)] = {
            "n_total":        n,
            "n_attack":       n_attack,
            "n_benign":       n_benign,
            "attack_frac":    n_attack / n,
            "purity":         purity,
        }
    return results
