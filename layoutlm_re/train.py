import os
import csv
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The `device` argument is deprecated and will be removed in v5 of Transformers", category=FutureWarning)
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from torch.amp import GradScaler, autocast

import config
from dataset import VRDDataset
from model import LayoutLMv3ForRelationExtraction
from utils import compute_metrics, apply_rsf

torch.cuda.empty_cache()

def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(epoch, model, dataloader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0

    optimizer.zero_grad() # Zero gradients at the beginning of the epoch or before the first accumulation

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for i, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bbox = batch['bbox'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        relation_matrix = batch['relation_matrix'].to(device)
        entity_token_indices = batch['entity_token_indices'].to(device)

        with autocast(config.DEVICE.type):
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                entity_token_indices=entity_token_indices,
                relation_matrix=relation_matrix
            )

        if loss is not None:
            current_loss_val = loss
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

            total_loss += current_loss_val.cpu().item()


        progress_bar.set_postfix({'loss': current_loss_val.item() if loss is not None else 0.0})

    # After the loop, if there are any remaining gradients (i.e., dataset size % ACCUMULATION_STEPS != 0)
    # For simplicity, this is often skipped, but for full correctness:
    if len(dataloader) % config.ACCUMULATION_STEPS != 0:
        if config.GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss


def evaluate_epoch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_eval_loss = 0
    valid_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            relation_matrix = batch['relation_matrix'].to(device)
            entity_token_indices = batch['entity_token_indices'].to(device)

            with autocast(config.DEVICE.type):
                loss, re_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    entity_token_indices=entity_token_indices,
                    relation_matrix=relation_matrix
                )

            total_eval_loss += loss
            valid_batches += 1

            if re_logits is not None and torch.isfinite(re_logits).all():
                pred_probs = torch.sigmoid(re_logits.float().clamp(-10, 10))  # Clamp for numerical stability

                for b_idx in range(pred_probs.shape[0]):
                    num_entities_in_sample = batch['num_entities'][b_idx].item()
                    if num_entities_in_sample == 0:
                        continue

                    probs_sample = pred_probs[b_idx, :num_entities_in_sample, :num_entities_in_sample]
                    labels_sample = relation_matrix[b_idx, :num_entities_in_sample, :num_entities_in_sample]

                    # Skip samples with NaN values
                    if not torch.isfinite(probs_sample).all() or not torch.isfinite(labels_sample).all():
                        continue
                        
                    preds_sample = apply_rsf(probs_sample, tau=config.RSF_TAU)

                    all_preds.append(preds_sample.cpu()) # Move to CPU before appending to list
                    all_labels.append(labels_sample.cpu())


    avg_eval_loss = total_eval_loss / valid_batches if valid_batches > 0 else float('nan')

    if not all_preds:
         print("Warning: No predictions generated during evaluation.")
         return {"precision": 0.0, "recall": 0.0, "f1": 0.0}, avg_eval_loss

    try:
        final_preds = torch.cat([p.flatten() for p in all_preds])
        final_labels = torch.cat([l.flatten() for l in all_labels])
        metrics = compute_metrics(final_preds, final_labels)
    except (RuntimeError, ValueError) as e:
        print(f"Error computing metrics: {e}")
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return metrics, avg_eval_loss


def main():
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Load Data ---
    print("Loading data...")
    # Use smaller dummy datasets for faster testing
    train_dataset = VRDDataset(config.TRAIN_DATA_PATH, config.TRAIN_PATH, is_train=True)
    val_dataset = VRDDataset(config.VAL_DATA_PATH, config.VAL_PATH, is_train=False)


    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=None)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=None)
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # --- Initialize Model ---
    print("Initializing model...")
    model = LayoutLMv3ForRelationExtraction()
    model.to(config.DEVICE)

    # --- Optimizer and Scheduler ---
    print("Setting up optimizer...")
    # Use parameter groups with different learning rates for backbone vs. heads
    backbone_params = [p for n, p in model.named_parameters() if "layoutlm" in n]
    classifier_params = [p for n, p in model.named_parameters() if "layoutlm" not in n]
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': config.LEARNING_RATE / 10},  # Lower LR for pretrained params
        {'params': classifier_params, 'lr': config.LEARNING_RATE}  # Higher LR for new params
    ], weight_decay=config.WEIGHT_DECAY)

    # Add gradient clipping
    max_grad_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    num_optimizer_steps = (len(train_dataloader) // config.ACCUMULATION_STEPS) * config.EPOCHS
    if len(train_dataloader) % config.ACCUMULATION_STEPS != 0:
        num_optimizer_steps += config.EPOCHS

    # Use percentage-based warmup with linear decay
    num_warmup_steps = int(config.WARMUP_RATIO * num_optimizer_steps)
    print(f"Warmup steps: {num_warmup_steps}, Total steps: {num_optimizer_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
                                               num_training_steps=num_optimizer_steps)

    # --- GradScaler for AMP ---
    scaler = GradScaler('cuda', enabled=config.DEVICE.type == 'cuda')

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
        train_loss = train_epoch(epoch, model, train_dataloader, optimizer, scheduler, config.DEVICE, scaler)
        val_metrics, val_loss = evaluate_epoch(model, val_dataloader, config.DEVICE)

        val_f1 = val_metrics['f1']
        val_precision = val_metrics.get('precision', float('nan'))
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