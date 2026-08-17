# Fed-HetFD

Fed-HetFD is a federated learning method designed to address heterogeneous feature distribution shifts in non-IID environments.  
The proposed method learns compact representations of local data using an autoencoder and characterizes client-specific feature distributions.  
Jensen–Shannon Divergence (JSD) is employed to measure distributional similarity between clients.  
Based on these similarities, clients are clustered to identify groups with comparable feature distributions.  
The framework then performs cluster-aware federated collaboration to improve learning under heterogeneous data distributions.  
Fed-HetFD is evaluated for federated intrusion detection using the CIC-IoT-2023 and UNSW-NB15 datasets.  
The repository provides the source code and experimental configurations required to reproduce the results reported in the associated paper.
