import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import csv
import numpy as np
import random
from tqdm.auto import tqdm # Progress bar
from torch.cuda.amp import GradScaler, autocast # For AMP

import config
from dataset import VRDDataset, collate_fn_bbs # Import custom collate if using BBS
from model import LayoutLMv3ForRelationExtraction
from utils import compute_metrics, apply_rsf

def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    total_re_loss = 0
    total_ee_loss = 0 # If using EEF

    optimizer.zero_grad() # Zero gradients at the beginning of the epoch or before the first accumulation

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for i, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bbox = batch['bbox'].to(device)
        pixel_values = batch['pixel_values'].to(device) if config.USE_IMAGE else None
        relation_matrix = batch['relation_matrix'].to(device)
        entity_token_indices = batch['entity_token_indices'].to(device)
        ee_labels = batch['ee_labels'].to(device) if config.USE_EEF else None

        # Forward pass with autocast for AMP
        with autocast(enabled=config.DEVICE.type == 'cuda'): # only enable autocast for CUDA
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                entity_token_indices=entity_token_indices,
                ee_labels=ee_labels,
                relation_matrix=relation_matrix
            )

            loss = outputs['loss']
            re_loss = outputs['re_loss']
            ee_loss = outputs['ee_loss'] # Will be None if not USE_EEF

        if loss is not None:
            # Normalize loss for accumulation
            # We accumulate the original loss value for reporting
            current_loss_val = loss.item()
            current_re_loss_val = re_loss.item() if re_loss is not None else 0
            current_ee_loss_val = ee_loss.item() if config.USE_EEF and ee_loss is not None else 0

            loss = loss / config.ACCUMULATION_STEPS

            # Backward pass
            scaler.scale(loss).backward()

            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                if config.GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad() # Zero gradients for the next accumulation cycle

            total_loss += current_loss_val
            total_re_loss += current_re_loss_val
            if config.USE_EEF:
                total_ee_loss += current_ee_loss_val

        # Log progress (using accumulated original loss values for average calculation)
        # Calculate averages based on number of batches processed so far, not accumulation steps
        num_processed_items = (i + 1)
        if (i + 1) % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / num_processed_items
            avg_re_loss = total_re_loss / num_processed_items
            log_str = f"Batch {i+1}/{len(dataloader)} | Avg Loss: {avg_loss:.4f} | Avg RE Loss: {avg_re_loss:.4f}"
            if config.USE_EEF and total_ee_loss > 0 : # Check total_ee_loss to avoid division by zero if no EEF batches yet
                 avg_ee_loss = total_ee_loss / num_processed_items
                 log_str += f" | Avg EE Loss: {avg_ee_loss:.4f}"
            print(log_str)

        progress_bar.set_postfix({'loss': current_loss_val if loss is not None else 0.0})

    # After the loop, if there are any remaining gradients (i.e., dataset size % ACCUMULATION_STEPS != 0)
    # For simplicity, this is often skipped, but for full correctness:
    if len(dataloader) % config.ACCUMULATION_STEPS != 0:
        if config.GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        # scheduler.step() # Typically scheduler steps with optimizer steps
        optimizer.zero_grad()


    avg_epoch_loss = total_loss / len(dataloader)
    avg_epoch_re_loss = total_re_loss / len(dataloader)
    avg_epoch_ee_loss = total_ee_loss / len(dataloader) if config.USE_EEF and total_ee_loss > 0 else 0
    print(f"End of Training Epoch | Avg Loss: {avg_epoch_loss:.4f} | Avg RE Loss: {avg_epoch_re_loss:.4f}" + (f" | Avg EE Loss: {avg_epoch_ee_loss:.4f}" if config.USE_EEF and total_ee_loss > 0 else ""))
    return avg_epoch_loss


def evaluate_epoch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_eval_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            pixel_values = batch['pixel_values'].to(device) if config.USE_IMAGE else None
            relation_matrix = batch['relation_matrix'].to(device)
            entity_token_indices = batch['entity_token_indices'].to(device)
            ee_labels = batch['ee_labels'].to(device) if config.USE_EEF else None

            # Forward pass with autocast for AMP consistency
            with autocast(enabled=config.DEVICE.type == 'cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    entity_token_indices=entity_token_indices,
                    ee_labels=ee_labels, # Pass labels to calculate potential loss during eval
                    relation_matrix=relation_matrix
                )

            if outputs['loss'] is not None:
                 total_eval_loss += outputs['loss'].item()

            re_logits = outputs['re_logits']

            if re_logits is not None:
                pred_probs = torch.sigmoid(re_logits.float()) # Ensure float for ops if logits were fp16

                for b_idx in range(pred_probs.shape[0]):
                    num_entities_in_sample = batch['num_entities'][b_idx].item()
                    if num_entities_in_sample == 0:
                         continue

                    probs_sample = pred_probs[b_idx, :num_entities_in_sample, :num_entities_in_sample]
                    labels_sample = relation_matrix[b_idx, :num_entities_in_sample, :num_entities_in_sample]

                    if config.USE_RSF:
                        preds_sample = apply_rsf(probs_sample, tau=config.RSF_TAU)
                    else:
                        preds_sample = (probs_sample > 0.5).long()

                    all_preds.append(preds_sample.cpu()) # Move to CPU before appending to list
                    all_labels.append(labels_sample.cpu())


    avg_eval_loss = total_eval_loss / len(dataloader) if len(dataloader) > 0 else 0
    print(f"End of Evaluation | Avg Loss: {avg_eval_loss:.4f}")

    if not all_preds:
         print("Warning: No predictions generated during evaluation.")
         return {"precision": 0.0, "recall": 0.0, "f1": 0.0}, avg_eval_loss

    final_preds = torch.cat([p.flatten() for p in all_preds])
    final_labels = torch.cat([l.flatten() for l in all_labels])

    metrics = compute_metrics(final_preds, final_labels)
    print(f"Evaluation Metrics: {metrics}")
    return metrics, avg_eval_loss


def main():
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Initialize Tokenizer and Processor ---
    # (Assuming this is handled elsewhere or not strictly needed for this script part)

    # --- Load Data ---
    print("Loading data...")
    # Use smaller dummy datasets for faster testing
    train_dataset = VRDDataset(config.TRAIN_DATA_PATH, is_train=True)
    val_dataset = VRDDataset(config.VAL_DATA_PATH, is_train=False)

    collate_function = collate_fn_bbs if config.USE_BBS else None # Or your default collate

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_function)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_function)
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # --- Initialize Model ---
    print("Initializing model...")
    model = LayoutLMv3ForRelationExtraction()
    model.to(config.DEVICE)

    # --- Optimizer and Scheduler ---
    print("Setting up optimizer...")
    if config.OPTIMIZER == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Optimizer {config.OPTIMIZER} not supported.")

    num_optimizer_steps = (len(train_dataloader) // config.ACCUMULATION_STEPS) * config.EPOCHS
    if len(train_dataloader) % config.ACCUMULATION_STEPS != 0:
        num_optimizer_steps += config.EPOCHS

    num_warmup_steps = int(0.1 * num_optimizer_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_optimizer_steps)

    # --- GradScaler for AMP ---
    scaler = GradScaler(enabled=config.DEVICE.type == 'cuda')

    # --- CSV Logging Setup ---
    csv_log_path = os.path.join(config.OUTPUT_DIR, "training_log.csv")
    csv_headers = [
        "Epoch", "Train Loss", "Val Loss",
        "Val F1", "Val Precision", "Val Recall" # Add other metrics if evaluate_epoch returns them
    ]
    # Write header to CSV file (or create it if it doesn't exist)
    with open(csv_log_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)
    print(f"Logging epoch results to {csv_log_path}")

    # --- Training Loop ---
    print("Starting training...")
    best_val_f1 = -1.0

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, config.DEVICE, scaler)
        val_metrics, val_loss = evaluate_epoch(model, val_dataloader, config.DEVICE)

        val_f1 = val_metrics['f1']
        val_precision = val_metrics.get('precision', float('nan')) # Use .get for safety if metric might be missing
        val_recall = val_metrics.get('recall', float('nan'))

        print(f"Epoch {epoch+1} Validation F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log to CSV
        current_epoch_data = [
            epoch + 1,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{val_f1:.4f}",
            f"{val_precision:.4f}",
            f"{val_recall:.4f}"
        ]
        with open(csv_log_path, 'a', newline='') as f_csv: # Open in append mode
            writer = csv.writer(f_csv)
            writer.writerow(current_epoch_data)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(config.OUTPUT_DIR, config.CHECKPOINT_FILENAME)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} (F1: {best_val_f1:.4f})")

    print("\nTraining finished.")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Full training log saved to {csv_log_path}")

if __name__ == "__main__":
    main()