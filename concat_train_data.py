#!/usr/bin/env python3

import argparse
import json
import numpy as np
from scipy.sparse import block_diag, csr_matrix, load_npz, save_npz, isspmatrix_csr, isspmatrix_coo
import os
from pathlib import Path

def load_data(dataset, index):
    """Load individual dataset components."""
    dataset_dir = os.path.join(dataset, f'design_{index}')
    adj_path = os.path.join(dataset_dir, f'adj_full.npz')
    features_path = os.path.join(dataset_dir, f'feat_full.npy')
    trojan_map_path = os.path.join(dataset_dir, f'class_map.json')
    
    adj = load_npz(adj_path)
    features = np.load(features_path)
    with open(trojan_map_path, 'r') as f:
        trojan_map = json.load(f)
    
    return adj, features, trojan_map

def main():
    parser = argparse.ArgumentParser(description='Concatenate training data for GCN model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--data_spec', type=str, required=True, help='Path to data specification JSON file')
    parser.add_argument('--dataset_size', type=int, required=True, help='Number of datasets to process')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load data specification
    with open(args.data_spec, 'r') as f:
        data_spec = json.load(f)
    
    # Initialize data structures
    adj_full = csr_matrix((0, 0))
    adj_train = csr_matrix((0, 0))
    feats = np.array([])
    class_map = {}
    role = {
        "tr": [],
        "va": [],
        "te": []
    }
    
    node_index = 0
    
    # Process each dataset
    for i in range(args.dataset_size):
        print(f"Processing dataset {i}...")

        if i in data_spec["sk"]:
            continue
        
        # Load individual dataset components
        adj_matrix, features, trojan_map = load_data(args.dataset, i)

        # please check if there is any nan in adj_matrix and features, is nan print out
        
        if np.isnan(features).any():
            print("nan found in features")
            continue

        
        # Concatenate features
        if feats.size == 0:
            feats = features
        else:
            feats = np.vstack((feats, features))
        
        # Update adjacency matrices
        if adj_full.shape == (0, 0):
            adj_full = adj_matrix
        else:
            adj_full = block_diag((adj_full, adj_matrix))
        
        # Update training adjacency matrix
        if i in data_spec["tr"]:
            if adj_train.shape == (0, 0):
                adj_train = adj_matrix
            else:
                adj_train = block_diag((adj_train, adj_matrix))
        else:
            zero_matrix = csr_matrix(adj_matrix.shape)
            if adj_train.shape == (0, 0):
                adj_train = zero_matrix
            else:
                adj_train = block_diag((adj_train, zero_matrix))
        
        # Update class map and roles
        num_indices = adj_matrix.shape[0]
        for idx in range(num_indices):
            class_map[node_index] = trojan_map[str(idx)]
            if i in data_spec["tr"]:
                role["tr"].append(node_index)
            elif i in data_spec["va"]:
                role["va"].append(node_index)
            else:
                role["te"].append(node_index)
            node_index += 1
    
    # Save concatenated data
    print("Saving concatenated data...")
    adj_full = adj_full.tocsr()
    adj_train = adj_train.tocsr()
    save_npz(os.path.join(args.output, 'adj_full.npz'), adj_full)
    save_npz(os.path.join(args.output, 'adj_train.npz'), adj_train)
    np.save(os.path.join(args.output, 'feats.npy'), feats)
    
    with open(os.path.join(args.output, 'class_map.json'), 'w') as f:
        json.dump(class_map, f)
    
    with open(os.path.join(args.output, 'role.json'), 'w') as f:
        json.dump(role, f)
    
    print("Data concatenation completed successfully!")

    print("check adjecent matrix type")

    mat = load_npz(os.path.join(args.output, 'adj_full.npz'))

    print("Is CSR:", isspmatrix_csr(mat))  # ✅ Should be True
    print("Is COO:", isspmatrix_coo(mat))  # ❌ Should be False
    print("Type:", type(mat)) 

    mat = load_npz(os.path.join(args.output, 'adj_train.npz'))

    print("Is CSR:", isspmatrix_csr(mat))  # ✅ Should be True
    print("Is COO:", isspmatrix_coo(mat))  # ❌ Should be False
    print("Type:", type(mat)) 


if __name__ == "__main__":
    main() 