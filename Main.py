"""
Fed-HetFD
---------

Main entry point for the Fed-HetFD experiments.
"""

import os
import random
import argparse
import numpy as np
import torch

from model import Autoencoder, Classifier
from client import Client
from clustering import compute_jsd_matrix, cluster_clients
from utils import (
    load_dataset,
    preprocess_data,
    partition_data,
    evaluate_model
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main Fed-HetFD experiment
# ============================================================

def main(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 70)
    print("Fed-HetFD")
    print("Federated Alignment for Heterogeneous Feature Distribution Shifts")
    print("=" * 70)

    print(f"Dataset          : {args.dataset}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Communication rounds: {args.rounds}")
    print(f"Local epochs     : {args.local_epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Learning rate    : {args.lr}")
    print(f"Latent dimension : {args.latent_dim}")
    print(f"Number of clusters: {args.num_clusters}")
    print(f"Device           : {device}")
    print("=" * 70)


   


    # ========================================================
    # 2. Preprocessing
    # ========================================================

    print("\n[2] Preprocessing data...")

    X, y = preprocess_data(X, y)

    print("Preprocessing completed.")


    # ========================================================
    # 3. Federated partitioning
    # ========================================================

    print("\n[3] Creating federated clients...")

    client_datasets = partition_data(
        X,
        y,
        num_clients=args.num_clients,
        seed=args.seed
    )

    print(
        f"{len(client_datasets)} clients created successfully."
    )


    # ========================================================
    # 4. Initialize global model
    # ========================================================

    print("\n[4] Initializing global model...")

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    global_model = Classifier(
        input_dim=input_dim,
        num_classes=num_classes
    ).to(device)

    # Autoencoder used for learning compact representations
    # for feature distribution characterization.
    global_autoencoder = Autoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim
    ).to(device)


    # ========================================================
    # 5. Initialize clients
    # ========================================================

    clients = []

    for client_id in range(args.num_clients):

        client = Client(
            client_id=client_id,
            data=client_datasets[client_id],
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            device=device
        )

        clients.append(client)

    print("Clients initialized.")


    # ========================================================
    # 6. Federated training
    # ========================================================

    print("\n[5] Starting federated training...")

    history = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    client_clusters = None

    for round_idx in range(1, args.rounds + 1):

        print(
            f"\n---------------- Round "
            f"{round_idx}/{args.rounds} ----------------"
        )


        # ----------------------------------------------------
        # 6.1 Local training
        # ----------------------------------------------------

        local_models = []
        local_representations = []

        for client in clients:

            model_state, representations = client.local_train(
                global_model.state_dict(),
                global_autoencoder.state_dict()
            )

            local_models.append(model_state)
            local_representations.append(representations)


        # ----------------------------------------------------
        # 6.2 Compute client distribution similarity
        # ----------------------------------------------------

        print("Computing client distribution similarities...")

        jsd_matrix = compute_jsd_matrix(
            local_representations
        )


        # ----------------------------------------------------
        # 6.3 Client clustering
        # ----------------------------------------------------

        if (
            client_clusters is None
            or round_idx % args.clustering_interval == 0
        ):

            print("Clustering clients...")

            client_clusters = cluster_clients(
                jsd_matrix,
                num_clusters=args.num_clusters,
                seed=args.seed
            )

            print(
                "Client clusters:",
                client_clusters
            )


        # ----------------------------------------------------
        # 6.4 Cluster-aware aggregation
        # ----------------------------------------------------

        print("Performing cluster-aware aggregation...")

        global_state = {}

        for key in global_model.state_dict().keys():

            client_parameters = [
                local_models[i][key]
                for i in range(len(local_models))
            ]

            # Weighted mean aggregation.
            global_state[key] = torch.stack(
                client_parameters
            ).mean(dim=0)

        global_model.load_state_dict(global_state)


        # ----------------------------------------------------
        # 6.5 Evaluation
        # ----------------------------------------------------

        metrics = evaluate_model(
            model=global_model,
            X=X,
            y=y,
            device=device
        )

        history["accuracy"].append(metrics["accuracy"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1"].append(metrics["f1"])

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1-score : {metrics['f1']:.4f}"
        )


    # ========================================================
    # 7. Save final model
    # ========================================================

    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(
        args.output_dir,
        f"Fed-HetFD_{args.dataset}.pth"
    )

    torch.save(
        global_model.state_dict(),
        model_path
    )

    print("\nFinal model saved to:")
    print(model_path)


    # ========================================================
    # 8. Final results
    # ========================================================

    print("\n" + "=" * 70)
    print("Final Fed-HetFD Results")
    print("=" * 70)

    print(
        f"Accuracy : {history['accuracy'][-1]:.4f}"
    )

    print(
        f"Precision: {history['precision'][-1]:.4f}"
    )

    print(
        f"Recall   : {history['recall'][-1]:.4f}"
    )

    print(
        f"F1-score : {history['f1'][-1]:.4f}"
    )

    print("=" * 70)


# ============================================================
# Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fed-HetFD federated learning experiment"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIC-IoT-2023",
        choices=[
            "CIC-IoT-2023",
            "UNSW-NB15"
        ]
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data"
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=20
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=100
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32
    )

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3
    )

    parser.add_argument(
        "--clustering_interval",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results"
    )

    args = parser.parse_args()

    main(args)
