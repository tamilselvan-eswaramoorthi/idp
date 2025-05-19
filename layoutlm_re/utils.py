import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import config

def normalize_bbox(bbox, width, height):
    """Normalize bounding box coordinates to [0, 1000]."""
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def unnormalize_bbox(bbox, width, height):
    """Unnormalize bounding box coordinates from [0, 1000] to image scale."""
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]

def simplify_bbox(bbox):
    return [min(bbox[0::2]), min(bbox[1::2]), max(bbox[0::2]), max(bbox[1::2])]


def get_entity_token_indices(word_ids, entity_word_indices):
    """Maps entity spans (word indices) to first token indices."""
    entity_first_token_indices = []
    current_word_idx = -1
    word_to_first_token = {}

    if not word_ids:
        return []

    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id != current_word_idx:
                word_to_first_token[word_id] = token_idx
                current_word_idx = word_id

    for entity_span in entity_word_indices:
        first_word_index = entity_span[0]
        if first_word_index in word_to_first_token:
            entity_first_token_indices.append(word_to_first_token[first_word_index])
        else:
            # Handle cases where the entity's first word might be truncated/missing
             entity_first_token_indices.append(-1) # Indicate missing/invalid
            #  print(f"Warning: Could not find first token for entity starting at word index {first_word_index}")

    return entity_first_token_indices

def get_entity_embeddings(sequence_output, entity_token_indices, pooling="first"):
    """
    Extracts entity embeddings from token sequence output.

    Args:
        sequence_output (torch.Tensor): Token embeddings from a model.
                                        Shape: (batch_size, seq_len, hidden_size)
        entity_token_indices (list, torch.Tensor, or None):
            Container for entity specifications.
            - If None: implies no entities.
            - If list:
                - If pooling == "first": List of token indices. Each item can be an int,
                                         a 0D tensor, or a 1D tensor with one element.
                                         -1 indicates padding.
                - If pooling == "mean": List of (start, end_exclusive) tuples/lists.
                                        Each start/end can be int or compatible tensor.
                                        (-1, -1) indicates padding.
            - If torch.Tensor:
                - If pooling == "first":
                    - 0D or 1D tensor: Indices are shared across the batch.
                        - 0D: e.g., `torch.tensor(1)` -> one entity, index 1 for all batch items.
                        - 1D: e.g., `torch.tensor([1, 2])` -> two entities, indices 1 & 2 for all batch items.
                    - 2D tensor (batch_size, num_entities_per_item): Indices are per batch item.
                - If pooling == "mean": Must be a 2D tensor of shape (num_entities, 2)
                                        (spans shared across batch) or an empty 1D tensor.
        pooling (str): Pooling strategy. "first" or "mean". Default is "first".

    Returns:
        torch.Tensor: Entity embeddings. Shape: (batch_size, num_entities, hidden_size).
                      If no entities, shape is (batch_size, 0, hidden_size).
    """
    batch_size, seq_len, hidden_size = sequence_output.shape

    if pooling not in ["first", "mean"]:
        raise ValueError(f"Unsupported pooling strategy: {pooling}. Choose 'first' or 'mean'.")

    if entity_token_indices is None:
        return torch.empty(batch_size, 0, hidden_size,
                           device=sequence_output.device, dtype=sequence_output.dtype)

    # --- Handle Tensor Input ---
    if isinstance(entity_token_indices, torch.Tensor):
        # Ensure indices tensor is of long type for gather and direct indexing
        if not entity_token_indices.dtype == torch.long:
            # This conversion might truncate if original was float, but indices should be int.
            # Alternatively, raise an error if not long. For now, try to convert.
            try:
                entity_token_indices = entity_token_indices.long()
            except RuntimeError:
                raise TypeError(f"entity_token_indices tensor must be of an integer type or convertible to long. Got {entity_token_indices.dtype}")


        if pooling == "first":
            if entity_token_indices.ndim == 2:
                # Case: Per-batch entity indices (batch_size, num_entities_per_item)
                if entity_token_indices.shape[0] != batch_size:
                    raise ValueError(
                        f"For 'first' pooling with a 2D entity_token_indices tensor, "
                        f"the first dimension ({entity_token_indices.shape[0]}) must match "
                        f"the batch_size of sequence_output ({batch_size})."
                    )
                num_entities_per_item = entity_token_indices.shape[1]
                if num_entities_per_item == 0:
                    return torch.empty(batch_size, 0, hidden_size,
                                       device=sequence_output.device, dtype=sequence_output.dtype)

                # Use torch.gather for efficient per-batch indexing
                indices_for_gather = entity_token_indices  # Shape (B, N_ENT)

                # Create a mask for valid indices (-1 is padding, OOB is invalid)
                valid_mask = (indices_for_gather >= 0) & (indices_for_gather < seq_len)  # (B, N_ENT)

                # Prepare safe indices for gather: map invalid ones to 0 (will be masked out)
                safe_indices = indices_for_gather.clone()
                safe_indices[~valid_mask] = 0  # Use a valid index like 0 for invalid spots

                # Expand safe_indices for gather operation
                # Target shape for indices in gather: (B, N_ENT, H) if input is (B, S, H) and dim=1
                safe_indices_expanded = safe_indices.unsqueeze(2).expand(-1, -1, hidden_size)

                # Gather embeddings
                gathered_embeddings = torch.gather(sequence_output, 1, safe_indices_expanded)

                # Apply mask to zero out embeddings from invalid/padding indices
                final_embeddings = gathered_embeddings * valid_mask.unsqueeze(2).to(gathered_embeddings.dtype)
                return final_embeddings

            elif entity_token_indices.ndim == 1: # Indices shared across batch (N_ENT_SHARED,)
                num_entities = entity_token_indices.shape[0]
                iterator_source = entity_token_indices
            elif entity_token_indices.ndim == 0: # Single index, shared across batch
                num_entities = 1
                iterator_source = entity_token_indices # Loop will handle .item()
            else:
                raise ValueError(
                    f"For 'first' pooling, if entity_token_indices is a tensor, "
                    f"it must be 0D, 1D, or 2D. Got {entity_token_indices.ndim}D tensor "
                    f"with shape {entity_token_indices.shape}."
                )
            
            if num_entities == 0: # For 0D/1D cases if tensor was empty
                 return torch.empty(batch_size, 0, hidden_size,
                                   device=sequence_output.device, dtype=sequence_output.dtype)
            entity_source_is_list = False # Indicates tensor source for the loop below

        elif pooling == "mean": # Spans shared across batch
            if entity_token_indices.ndim == 2 and entity_token_indices.shape[1] == 2:
                num_entities = entity_token_indices.shape[0]
            elif entity_token_indices.ndim == 1 and entity_token_indices.numel() == 0: # Empty 1D tensor
                num_entities = 0
            else:
                raise ValueError(
                    f"For 'mean' pooling, if entity_token_indices is a tensor, "
                    f"it must be 2D with shape (num_entities, 2) or an empty 1D tensor. "
                    f"Got tensor with shape {entity_token_indices.shape}."
                )
            if num_entities == 0:
                 return torch.empty(batch_size, 0, hidden_size,
                                   device=sequence_output.device, dtype=sequence_output.dtype)
            iterator_source = entity_token_indices
            entity_source_is_list = False
    
    # --- Handle List Input or Fall-through from 0D/1D Tensor for 'first' or 2D for 'mean' ---
    elif isinstance(entity_token_indices, list):
        if not entity_token_indices:
            return torch.empty(batch_size, 0, hidden_size,
                               device=sequence_output.device, dtype=sequence_output.dtype)
        iterator_source = entity_token_indices
        num_entities = len(iterator_source)
        entity_source_is_list = True
    else:
        raise TypeError(
            f"entity_token_indices is expected to be a Python list, a torch.Tensor, or None. "
            f"Got type: {type(entity_token_indices)}."
        )

    # If num_entities somehow became 0 and wasn't caught (should be caught above)
    if num_entities == 0:
         return torch.empty(batch_size, 0, hidden_size,
                           device=sequence_output.device, dtype=sequence_output.dtype)

    # --- Main Processing Loop (for list inputs, or specific tensor cases) ---
    entity_embeds = []
    zero_embed_prototype = torch.zeros(batch_size, hidden_size,
                                       device=sequence_output.device, dtype=sequence_output.dtype)

    for i in range(num_entities):
        item_from_iterator = None
        if entity_source_is_list:
            item_from_iterator = iterator_source[i]
        else: # Source is Tensor (and not the 2D 'first' pooling case which returned early)
            if pooling == "first": # 0D or 1D tensor
                if iterator_source.ndim == 0:
                    item_from_iterator = iterator_source # The scalar tensor itself
                else: # 1D tensor
                    item_from_iterator = iterator_source[i]  # This will be a 0D tensor (scalar)
            elif pooling == "mean":  # iterator_source is 2D (N_ENT_SHARED, 2) tensor
                item_from_iterator = iterator_source[i, :]  # This will be a 1D tensor of shape (2,)

        # "first" pooling logic (for list items or 0D/1D tensor items)
        if pooling == "first":
            entity_idx_spec = item_from_iterator
            current_scalar_idx = None
            is_valid_idx_spec = True

            if isinstance(entity_idx_spec, torch.Tensor):
                if entity_idx_spec.numel() == 1:
                    current_scalar_idx = entity_idx_spec.item()
                else: is_valid_idx_spec = False
            elif isinstance(entity_idx_spec, int):
                current_scalar_idx = entity_idx_spec
            else: is_valid_idx_spec = False

            if is_valid_idx_spec and \
               current_scalar_idx is not None and \
               current_scalar_idx != -1 and \
               0 <= current_scalar_idx < seq_len:
                entity_embeds.append(sequence_output[:, current_scalar_idx, :])
            else:
                entity_embeds.append(zero_embed_prototype.clone())

        # "mean" pooling logic (for list items or 2D shared tensor items)
        elif pooling == "mean":
            span_spec = item_from_iterator
            start_idx_processed, end_idx_processed = -1, -1
            is_valid_span_spec = True
            s_orig, e_orig = None, None

            if isinstance(span_spec, (list, tuple)) and len(span_spec) == 2:
                s_orig, e_orig = span_spec[0], span_spec[1]
            elif isinstance(span_spec, torch.Tensor) and \
                 span_spec.ndim == 1 and span_spec.shape[0] == 2:
                s_orig, e_orig = span_spec[0], span_spec[1] # s_orig, e_orig are 0D Tensors
            else: is_valid_span_spec = False

            if is_valid_span_spec:
                temp_s = s_orig
                if isinstance(s_orig, torch.Tensor):
                    if s_orig.numel() == 1: temp_s = s_orig.item()
                    else: is_valid_span_spec = False
                if not isinstance(temp_s, int): is_valid_span_spec = False
                else: start_idx_processed = temp_s

                if is_valid_span_spec:
                    temp_e = e_orig
                    if isinstance(e_orig, torch.Tensor):
                        if e_orig.numel() == 1: temp_e = e_orig.item()
                        else: is_valid_span_spec = False
                    if not isinstance(temp_e, int): is_valid_span_spec = False
                    else: end_idx_processed = temp_e
            
            if not is_valid_span_spec:
                entity_embeds.append(zero_embed_prototype.clone())
            elif start_idx_processed == -1 and end_idx_processed == -1:
                entity_embeds.append(zero_embed_prototype.clone())
            elif 0 <= start_idx_processed < end_idx_processed <= seq_len:
                entity_span_tokens = sequence_output[:, start_idx_processed:end_idx_processed, :]
                if entity_span_tokens.shape[1] == 0: pooled_embed = zero_embed_prototype.clone()
                else: pooled_embed = torch.mean(entity_span_tokens, dim=1)
                entity_embeds.append(pooled_embed)
            else:
                entity_embeds.append(zero_embed_prototype.clone())

    return torch.stack(entity_embeds, dim=1)

def apply_rsf(pred_probs, tau=0.1):
    """Applies Restriction on Selection of Fathers post-processing."""
    # pred_probs shape: (num_entities, num_entities) - Assuming single batch item for simplicity here
    num_entities = pred_probs.shape[0]
    if num_entities == 0:
        return torch.zeros_like(pred_probs, dtype=torch.long)

    final_preds = torch.zeros_like(pred_probs, dtype=torch.long)
    max_probs, _ = torch.max(pred_probs, dim=0) # Max probability for each child (column j)

    for i in range(num_entities):
        for j in range(num_entities):
            if i == j: continue # No self-loops typically
            prob_ij = pred_probs[i, j]
            if prob_ij > 0.5 and prob_ij >= (max_probs[j] - tau):
                # Consider j as a parent of i only if prob > 0.5 AND it's the max (within margin tau)
                # Note: The paper description seems slightly ambiguous:
                # "j-th entity is considered as a parent entity of the i-th entity if P_i->j > 0.5 and P_i->j is the maximum..."
                # This implies finding the max parent for each child 'i'.
                # Let's implement finding the single best parent for child 'i' based on the paper's formula notation Rij = 1(Pi->j > 0.5) * 1(max_k Pk->j < Pi->j + tau)
                # This seems to check if 'i' is the best parent *for* 'j'. Let's stick to the paper's formula interpretation.

                # Check if i is the max parent for j within margin tau
                is_max_parent = True
                for k in range(num_entities):
                     if k != i and pred_probs[k, j] > prob_ij + tau:
                         is_max_parent = False
                         break
                if is_max_parent:
                     final_preds[i, j] = 1 # Predict link from i to j
            # The original interpretation (find best parent for each child 'i') might be:
            # max_parent_prob_for_i, _ = torch.max(pred_probs[:, i])
            # if prob_ji > 0.5 and prob_ji >= max_parent_prob_for_i - tau:
            #     final_preds[j, i] = 1 # Predict link from j to i

    return final_preds

def compute_metrics(predictions, labels):
    """Computes Precision, Recall, F1 for the relation matrix."""
    # Flatten the matrices
    predictions = predictions.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    # Ensure there are positive predictions or labels to avoid division by zero warnings
    if np.sum(predictions) == 0 and np.sum(labels) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    elif np.sum(labels) == 0: # No true positives possible, only FP or TN
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0} if np.sum(predictions) > 0 else {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    elif np.sum(predictions) == 0: # No predicted positives, only FN or TN
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0} # Recall is 0 if there were true labels

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

# TODO: Implement compute_variance_loss if USE_RSF_LOSS is True