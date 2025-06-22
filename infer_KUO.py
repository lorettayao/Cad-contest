import torch
import numpy as np
from scipy.sparse import load_npz
import argparse
import os
import json
import yaml
from sklearn.preprocessing import StandardScaler
from infer_model import GraphSAINT
from infer_utils import *
import torch

def coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))

def load_config(train_config_path):
    with open(train_config_path) as f:
        config = yaml.safe_load(f)
    arch_gcn = {
        'dim': -1,
        'aggr': 'concat',
        'loss': 'softmax',
        'arch': '1',
        'act': 'I',
        'bias': 'norm'
    }
    arch_gcn.update(config['network'][0])
    train_params = {
        'lr': 1e-4,
        'weight_decay': 0.0,
        'norm_loss': True,
        'norm_aggr': True,
        'q_threshold': 50,
        'q_offset': 0
    }
    train_params.update(config['params'][0])
    return arch_gcn, train_params

def run_inference(model_path, feats_path, adj_path, pred_path, train_config, device='cpu'):
    # Load features and adjacency
    # adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    # feats = np.load('./{}/feats.npy'.format(prefix))

    adj = load_npz(adj_path).astype(np.int32)
    normalized_adj = adj_norm(adj).tocoo()
    feats_npz = np.load(feats_path)
    feats = feats_npz['feats']  # 取出實際的 ndarray
    scaler = StandardScaler()
    scaler.fit(feats)
    feats = scaler.transform(feats)
    feats_tensor = torch.tensor(feats, dtype=torch.float32).to(device)

    # coo to torch
    adj_tensor = coo_scipy2torch(normalized_adj)
    node_graph = np.arange(normalized_adj.shape[0])
    norm_loss_test = np.ones(normalized_adj.shape[0]) / len(node_graph)
    print("norm_loss_test: ", norm_loss_test)
    norm_loss_test = torch.tensor(norm_loss_test, dtype=torch.float32).to(device)

    # Debug information for input data
    print("\n=== Input Data Statistics ===")
    print("Features shape:", feats.shape)
    print("Non-zero features:", np.count_nonzero(feats))
    
    print("\nAdjacency matrix shape:", adj.shape)
    print("Adjacency non-zero elements:", adj.nnz)
    print("Average node degree:", adj.sum() / adj.shape[0])
    print("Max node degree:", np.max(adj.sum(axis=1)))

    
    # Load architecture and training params
    arch_gcn, train_params = load_config(train_config)
    num_classes = arch_gcn.get("num_classes", 2)  # fallback
    label_dummy = np.zeros((feats.shape[0], num_classes))  # dummy labels

    print("\n=== Model Configuration ===")
    print("Architecture:", arch_gcn)
    print("Training params:", train_params)
    print("Number of classes:", num_classes)

    # Rebuild model
    model = GraphSAINT(
        num_classes=num_classes,
        arch_gcn=arch_gcn,
        train_params=train_params,
        feat_full=feats,
        label_full=label_dummy,
        cpu_eval=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    loss, preds, labels = model.eval_step(node_graph, adj_tensor, norm_loss_test)
    print("loss: ", loss)
    print("preds: ", preds)
    preds_bin = preds.numpy()
    if num_classes == 2:
        preds_bin = preds_bin[:,1] >= preds_bin[:,0]
        print("\n=== Binary Predictions ===")
        print("Class 0 count:", np.sum(preds_bin == 0))
        print("Class 1 count:", np.sum(preds_bin == 1))

    np.save(pred_path, preds)
    print(f"\n[✓] Saved predictions to {pred_path}")
    return preds.numpy()

def generate_iccad_output(preds, gate_to_index, output_path="output_result.txt", threshold=0.5, multiclass=True):
    """
    Generate ICCAD output format.
    
    Args:
        preds: numpy array of predictions
        gate_to_index: dict mapping gate names to their indices
        output_path: path to save the ICCAD output
        threshold: threshold for binary classification
        multiclass: whether this is a multiclass problem
    """
    if multiclass:
        preds_bin = np.argmax(preds, axis=1)
    else:
        preds_bin = (preds > threshold).astype(int)

    trojan_gate_list = [
        gate_name
        for idx, gate_name in gate_to_index.items()
        if preds_bin[int(idx)] == 1
    ]

    with open(output_path, 'w') as f:
        if len(trojan_gate_list) >= 15
            f.write("TROJANED\nTROJAN_GATES\n")
            for gate in trojan_gate_list:
                f.write(f"{gate}\n")
            f.write("END_TROJAN_GATES\n")
        else:
            f.write("NO_TROJAN\n")

    print(f"[✓] ICCAD result written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on a trained GraphSAINT model')
    parser.add_argument('--model', type=str, required=True, 
                      help='Path to the trained model file (e.g., saved_model.pkl)')
    parser.add_argument('--features', type=str, required=True, 
                      help='Path to the node features file (e.g., *_feats.npy)')
    parser.add_argument('--adjacency', type=str, required=True, 
                      help='Path to the adjacency matrix file (e.g., *_adj_full.npz)')
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to the model configuration file (e.g., train_config.yaml)')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save the prediction results (e.g., predictions.npy)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                      help='Classification threshold for binary classification')
    parser.add_argument('--gate_mapping', type=str, required=False,
                      help='Path to JSON file containing gate-to-index mapping')
    parser.add_argument('--iccad_output', type=str, required=False,
                      help='Generate ICCAD output')
    args = parser.parse_args()

    # Run inference
    preds = run_inference(
        args.model,
        args.features,
        args.adjacency,
        args.output,
        args.config,
        args.device
    )
    
    # Generate ICCAD output if needed
    if args.iccad_output:
        
        if not args.gate_mapping:
            raise ValueError("--gate_mapping is required when --iccad_output is specified")

        # Load gate-to-index mapping
        with open(args.gate_mapping) as f:
            gate_to_index = json.load(f)
        
        # Generate ICCAD output
        generate_iccad_output(preds, gate_to_index, args.iccad_output, args.threshold)

if __name__ == '__main__':
    main()
