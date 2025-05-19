# predict.py

import torch
import os
import json
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Assuming these files are in the same directory or accessible via PYTHONPATH
import config # Main configuration file
from dataset import VRDDataset, collate_fn_bbs # Ensure VRDDataset can handle inference data
from model import LayoutLMv3ForRelationExtraction
from utils import apply_rsf # If USE_RSF is True
from train import set_seed # Re-use set_seed from train.py

def predict_relations(model, dataloader, device, output_file_path, cfg):
    """
    Generates relation predictions for the given data and saves them to a JSON file.

    Args:
        model: The trained relation extraction model.
        dataloader: DataLoader for the prediction dataset.
        device: The device to run inference on (e.g., 'cuda', 'cpu').
        output_file_path: Path to save the JSON predictions.
        cfg: The configuration object (e.g., loaded from config.py and potentially updated).
    """
    model.eval()
    all_document_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Predicting Relations", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            
            pixel_values = batch.get('pixel_values') # Use .get for safety
            if pixel_values is not None and cfg.USE_IMAGE:
                pixel_values = pixel_values.to(device)
            else:
                pixel_values = None # Ensure it's None if not used or not available in batch

            entity_token_indices = batch['entity_token_indices'].to(device)
            # For prediction, ee_labels and relation_matrix are not passed to model.forward()

            # Forward pass with autocast for AMP consistency if model was trained with it
            with autocast(enabled=(device.type == 'cuda' and cfg.USE_AMP_INFERENCE if hasattr(cfg, 'USE_AMP_INFERENCE') else device.type == 'cuda')):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    entity_token_indices=entity_token_indices
                    # ee_labels and relation_matrix default to None in model's forward
                )

            re_logits = outputs.get('re_logits') # Shape: (batch_size, num_entities, num_entities)

            if re_logits is None:
                print(f"Warning: 're_logits' not found in model output for batch {batch_idx}. Skipping batch.")
                continue

            # Ensure logits are float for sigmoid, if they were e.g. fp16 from autocast
            pred_probs = torch.sigmoid(re_logits.float())

            # Process each sample in the batch
            for i in range(pred_probs.shape[0]):
                # Dataset should return 'ids' for sample identification and 'num_entities'
                # Fallback for 'ids' if not present (though it's highly recommended)
                sample_id_list = batch.get('ids')
                if sample_id_list is None: # Fallback
                    sample_id = f"batch{batch_idx}_sample{i}"
                else:
                    sample_id = sample_id_list[i]
                
                num_entities_in_sample = batch['num_entities'][i].item()

                sample_prediction = {
                    "id": sample_id,
                    "relations": []
                    # Consider adding "entities": [...] list here if you want to store
                    # entity texts/bboxes from the input data alongside predictions.
                    # This would require VRDDataset to pass through original entity info.
                }

                if num_entities_in_sample == 0:
                    all_document_predictions.append(sample_prediction)
                    continue

                # Extract probabilities for the current sample, considering padding
                probs_sample = pred_probs[i, :num_entities_in_sample, :num_entities_in_sample]

                if cfg.USE_RSF:
                    # apply_rsf expects probabilities
                    preds_matrix = apply_rsf(probs_sample, tau=cfg.RSF_TAU)
                else:
                    # Standard thresholding for binary predictions
                    preds_matrix = (probs_sample > cfg.PREDICTION_THRESHOLD if hasattr(cfg, 'PREDICTION_THRESHOLD') else probs_sample > 0.5).long()

                for head_idx in range(num_entities_in_sample):
                    for tail_idx in range(num_entities_in_sample):
                        # Control whether self-relations are allowed via config
                        if head_idx == tail_idx and not (hasattr(cfg, 'ALLOW_SELF_RELATIONS') and cfg.ALLOW_SELF_RELATIONS):
                            continue
                        
                        if preds_matrix[head_idx, tail_idx].item() == 1:
                            sample_prediction["relations"].append({
                                "head_entity_idx": head_idx, # Index of head entity within this sample's entity list
                                "tail_entity_idx": tail_idx, # Index of tail entity
                                "score": probs_sample[head_idx, tail_idx].item() # Predicted probability
                            })
                
                all_document_predictions.append(sample_prediction)

    # Save all predictions to a single JSON file
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir: # If output_file_path includes a directory
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file_path, 'w') as f:
        json.dump(all_document_predictions, f, indent=2)
    print(f"Predictions saved to {output_file_path}")

    return all_document_predictions


def main():
    parser = argparse.ArgumentParser(description="Predict relations using a trained LayoutLMv3 model.")
    parser.add_argument("--model_checkpoint", type=str, default = 'output/best_model.pt'
                        help="Path to the trained model checkpoint (.pt or .bin).")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input data file (e.g., JSON/CSV file for VRDDataset).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the prediction results (JSON file). "
                             "Defaults to 'predictions.json' in config.OUTPUT_DIR.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for prediction. Overrides config.BATCH_SIZE if provided.")
    # Add other relevant config overrides if needed, e.g.:
    # parser.add_argument("--use_image", type=bool, default=None, help="Override config.USE_IMAGE.")
    # parser.add_argument("--use_rsf", type=bool, default=None, help="Override config.USE_RSF.")

    args = parser.parse_args()

    # --- Configuration Setup ---
    # Create a mutable copy of config settings if you plan to modify them
    # For simplicity, we'll directly use/update 'config' if it's a module,
    # or pass a dictionary if config were structured differently.
    # Here, 'config' is treated as the global config module.
    
    # Override config with command-line arguments if provided
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    # Example for other overrides:
    # if args.use_image is not None:
    #     config.USE_IMAGE = args.use_image
    # if args.use_rsf is not None:
    #     config.USE_RSF = args.use_rsf

    output_file_path = args.output_file
    if output_file_path is None:
        # Ensure config.OUTPUT_DIR exists if we're using it for the default output path
        if hasattr(config, 'OUTPUT_DIR') and config.OUTPUT_DIR:
             os.makedirs(config.OUTPUT_DIR, exist_ok=True)
             output_file_path = os.path.join(config.OUTPUT_DIR, "predictions.json")
        else: # Fallback if OUTPUT_DIR is not defined or empty
            output_file_path = "predictions.json"


    # --- Reproducibility and Device ---
    set_seed(config.SEED)
    
    # Determine device, ensuring cfg.DEVICE respects CUDA availability
    if hasattr(config, 'DEVICE') and config.DEVICE == "cuda" and not torch.cuda.is_available():
        print("Warning: config.DEVICE is 'cuda' but CUDA is not available. Using 'cpu'.")
        device = torch.device("cpu")
    elif hasattr(config, 'DEVICE'):
        device = torch.device(config.DEVICE)
    else: # Default if not in config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # --- Load Data ---
    print(f"Loading data from {args.data_path}...")
    # VRDDataset should be flexible for inference (e.g., is_train=False might mean no labels)
    # It must return 'ids' (for sample identification) and 'num_entities'.
    # Also, 'entity_token_indices' are crucial.
    predict_dataset = VRDDataset(
        data_path=args.data_path,
        is_train=False # Crucial: ensures dataset behaves for inference/test mode
                       # (e.g., doesn't expect labels, might have different tokenization settings)
    )
    
    collate_function = collate_fn_bbs if config.USE_BBS else None # Or your project's default collate_fn

    predict_dataloader = DataLoader(
        predict_dataset,
        batch_size=config.BATCH_SIZE, # Uses batch_size from config (potentially updated by args)
        shuffle=False,                # No shuffling for prediction
        collate_fn=collate_function,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 0
    )
    print(f"Loaded {len(predict_dataset)} samples for prediction.")

    # --- Initialize Model ---
    print("Initializing model...")
    # LayoutLMv3ForRelationExtraction initializes using values from config.py (e.g., MODEL_NAME, CACHE_DIR)
    model = LayoutLMv3ForRelationExtraction() 

    print(f"Loading model checkpoint from {args.model_checkpoint}...")
    # Load state dict; map_location ensures model tensors are moved to the correct device
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)

    # --- Perform Prediction ---
    print("Starting prediction run...")
        
    _ = predict_relations(model, predict_dataloader, device, output_file_path, config)

    print(f"Prediction finished. Results saved to {output_file_path}")

if __name__ == "__main__":
    # Ensure necessary config attributes have defaults if not present
    # This is a good practice for attributes used in predict_relations
    if not hasattr(config, 'USE_AMP_INFERENCE'):
        config.USE_AMP_INFERENCE = True # Default to True for CUDA, False for CPU (handled by autocast enabled flag)
    if not hasattr(config, 'ALLOW_SELF_RELATIONS'):
        config.ALLOW_SELF_RELATIONS = False
    if not hasattr(config, 'PREDICTION_THRESHOLD'):
        config.PREDICTION_THRESHOLD = 0.5
        
    main()