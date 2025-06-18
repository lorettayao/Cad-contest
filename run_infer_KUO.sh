design_name="design26"
timestamp="2025-06-19_03-14-51"
python3 infer_KUO.py \
    --model ./trained_model/${timestamp}/saved_model_${timestamp}.pkl \
    --features GraphSAINT/data/parsered_features/${design_name}/feat_full.npz \
    --adjacency GraphSAINT/data/parsered_features/${design_name}/adj_full.npz \
    --config trained_model/${timestamp}/DATE21.yml \
    --output ./output_infer_${design_name} \
    --gate_mapping GraphSAINT/data/parsered_features/${design_name}/gate_map.json \
    --iccad_output ./output_iccad_${design_name}.txt