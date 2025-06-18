design_name="design26"
python3 infer_KUO.py \
    --model ./trained_model/2025-06-18_16-11-57/saved_model_2025-06-18_16-11-57.pkl \
    --features GraphSAINT/data/parsered_features/${design_name}/feat_full.npz \
    --adjacency GraphSAINT/data/parsered_features/${design_name}/adj_full.npz \
    --config trained_model/2025-06-18_16-11-57/DATE21.yml \
    --output ./output_infer_${design_name} \
    --gate_mapping GraphSAINT/data/parsered_features/${design_name}/gate_map.json \
    --iccad_output ./output_iccad_${design_name}.txt