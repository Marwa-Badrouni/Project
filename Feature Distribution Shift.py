import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def federated_feature_shift_split(data, label_col, num_clients=5):
    """
    Simulates statistical heterogeneity (feature distribution shift)
    WITHOUT label distribution skew.

    Args:
        data: pandas DataFrame (features + label)
        label_col: name of label column
        num_clients: number of federated clients (K)

    Returns:
        dict: {client_id: DataFrame}
    """

    clients_data = {i: [] for i in range(num_clients)}

    labels = data[label_col].unique()

    scaler = MinMaxScaler()
    feature_cols = [col for col in data.columns if col != label_col]

    # Normalize features (important for fair score computation)
    data_scaled = data.copy()
    data_scaled[feature_cols] = scaler.fit_transform(data[feature_cols])

    # STEP 1: class-wise processing (IMPORTANT to avoid label skew)
    for c in labels:
        class_data = data_scaled[data_scaled[label_col] == c].copy()

        X = class_data[feature_cols].values

        # STEP 2: compute global statistical score
        score = np.mean(X, axis=1)

        class_data["score"] = score

        # STEP 3: sort by score
        class_data = class_data.sort_values(by="score")

        # STEP 4: split into K quantiles
        splits = np.array_split(class_data, num_clients)

        # STEP 5: assign each split to corresponding client
        for i in range(num_clients):
            clients_data[i].append(splits[i])

    # STEP 6: merge per client
    final_clients = {}
    for i in range(num_clients):
        final_clients[i] = pd.concat(clients_data[i]).drop(columns=["score"])

    return final_clients
