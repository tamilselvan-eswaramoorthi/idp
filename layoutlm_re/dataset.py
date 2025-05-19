import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor

import config
from utils import normalize_bbox, get_entity_token_indices

class VRDDataset(Dataset):
    def __init__(self, data_path, is_train=False):
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(config.MODEL_NAME, cache_dir=config.CACHE_DIR)
        self.image_processor = LayoutLMv3ImageProcessor(apply_ocr=False) # OCR is done externally
        self.image_dir = config.IMAGE_DIR
        self.max_length = config.MAX_LENGTH
        self.is_train = is_train
        self.pad_token_box = [0, 0, 0, 0]

        # Load data from jsonl file
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_dir, sample['image_path'])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # --- Text and Boxes ---
        words = sample['words']
        unnormalized_boxes = sample['bboxes']

        if config.USE_BBO:
            # ... (your BBO logic - ensure entities/relations are pre-adjusted in data)
            pass

        normalized_boxes = [normalize_bbox(box, width, height) for box in unnormalized_boxes]

        # --- Entities ---
        original_entities = sample['entities'] # List of dicts

        # Truncate entities if they exceed MAX_ENTITIES
        if len(original_entities) > config.MAX_ENTITIES:
            # print(f"Warning: Sample {idx} has {len(original_entities)} entities, truncating to {config.MAX_ENTITIES}.")
            entities = original_entities[:config.MAX_ENTITIES]
        else:
            entities = original_entities
        
        num_actual_entities_to_process = len(entities) # This will be <= config.MAX_ENTITIES

        entity_word_indices = [e['span'] for e in entities]
        entity_labels = [e['label'] for e in entities] # For EEF

        if config.USE_EM:
            # ... (your EM logic)
            # Make sure entity_word_indices are updated correctly AFTER potential entity truncation
            # This part might need careful re-evaluation if EM relies on original_entities indices
            # For simplicity, assume entity_word_indices are based on the (potentially truncated) `entities` list
            original_word_count = len(words)
            new_words = []
            new_boxes = []
            word_idx_map_for_em = {}
            current_new_idx = 0

            # Rebuild entity_starts based on the (potentially truncated) `entities` list
            entity_starts = {e['span'][0]: e['label'] for e in entities}

            for i in range(original_word_count):
                marker_insertion_point = current_new_idx
                if i in entity_starts:
                    entity_type = entity_starts[i].lower().replace("-", "_")
                    new_words.append(entity_type)
                    new_boxes.append(normalized_boxes[i])
                    current_new_idx += 1
                
                word_insertion_point = current_new_idx
                new_words.append(words[i])
                new_boxes.append(normalized_boxes[i])
                current_new_idx += 1
                word_idx_map_for_em[i] = [marker_insertion_point, word_insertion_point, current_new_idx]

            words = new_words
            normalized_boxes = new_boxes

            updated_entity_word_indices = []
            for ent_idx in range(num_actual_entities_to_process): # Iterate over (truncated) entities
                original_span_start, original_span_end = entities[ent_idx]['span'] # Use span from current entity
                # Ensure original_span_start and original_span_end are valid keys for word_idx_map_for_em
                # This implies that the spans in `entities` are still valid indices for the original `words` list
                # before EM modification. This is a critical assumption.
                if original_span_start in word_idx_map_for_em and original_span_end in word_idx_map_for_em:
                    new_word_span_start = word_idx_map_for_em[original_span_start][1]
                    new_word_span_end = word_idx_map_for_em[original_span_end][1]
                    updated_entity_word_indices.append([new_word_span_start, new_word_span_end])
                else:
                    # This case should be handled: what if an entity's span is outside the original words?
                    # Or if an entity was truncated and its original span is no longer relevant.
                    # For now, we might skip this entity or add a placeholder.
                    # Let's assume for now that spans are valid for words before EM.
                    print(f"Warning: Entity span issue during EM for sample {idx}, entity {ent_idx}. Skipping this entity's span update.")
                    # Potentially add a dummy span or handle this error. For now, this might lead to fewer entity spans.
                    pass # This could lead to a mismatch later if not handled robustly

            entity_word_indices = updated_entity_word_indices
            # After EM, entity_labels should still correspond to the entities in the (truncated) `entities` list
            # and entity_word_indices should also correspond one-to-one with this list.
            # Ensure len(entity_word_indices) == num_actual_entities_to_process


        encoding = self.tokenizer(
            words,
            boxes=normalized_boxes,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        image_features = self.image_processor(image, return_tensors="pt")
        encoding['pixel_values'] = image_features['pixel_values'].squeeze(0)

        # --- Relation Matrix ---
        # Use num_actual_entities_to_process which is capped by config.MAX_ENTITIES
        actual_relation_matrix = torch.zeros((num_actual_entities_to_process, num_actual_entities_to_process), dtype=torch.float)
        
        # entity_id_to_idx should be built from the (potentially truncated) `entities` list
        entity_id_to_idx = {e['id']: i for i, e in enumerate(entities)} # `entities` is already truncated

        if 'relations' in sample:
            for rel in sample['relations']:
                head_id = rel['head']
                tail_id = rel['tail']
                # Only consider relations where both head and tail entities are within the processed (truncated) set
                if head_id in entity_id_to_idx and tail_id in entity_id_to_idx:
                    head_idx = entity_id_to_idx[head_id]
                    tail_idx = entity_id_to_idx[tail_id]
                    # These indices are already relative to the truncated list, so they will be < num_actual_entities_to_process
                    actual_relation_matrix[head_idx, tail_idx] = 1.0
        
        padded_relation_matrix = torch.full(
            (config.MAX_ENTITIES, config.MAX_ENTITIES), 0.0, dtype=torch.float
        )
        if num_actual_entities_to_process > 0:
            # Now this assignment is safe because num_actual_entities_to_process <= config.MAX_ENTITIES
            padded_relation_matrix[:num_actual_entities_to_process, :num_actual_entities_to_process] = actual_relation_matrix
        encoding['relation_matrix'] = padded_relation_matrix

        # --- Entity Token Indices ---
        # entity_word_indices should correspond to the (truncated) `entities` list
        # and correctly mapped if EM was used.
        actual_entity_first_token_indices = get_entity_token_indices(word_ids, entity_word_indices)
        
        padded_entity_token_indices = torch.full(
            (config.MAX_ENTITIES,), config.ENTITY_INDEX_PAD_VALUE, dtype=torch.long
        )

        # Ensure that the number of token indices matches num_actual_entities_to_process before padding
        num_found_token_indices = len(actual_entity_first_token_indices)

        if num_actual_entities_to_process > 0:
            # Only copy if the lengths match, or take the minimum of the two
            # This handles cases where get_entity_token_indices might return fewer indices
            # (e.g., if some entities were truncated by the tokenizer itself, or EM issues)
            copy_len = min(num_actual_entities_to_process, num_found_token_indices)
            if copy_len > 0:
                 padded_entity_token_indices[:copy_len] = torch.tensor(actual_entity_first_token_indices[:copy_len], dtype=torch.long)
            if num_actual_entities_to_process != num_found_token_indices:
                print(f"Warning: Sample {idx}: Mismatch between processed entities ({num_actual_entities_to_process}) and found token indices ({num_found_token_indices}). Using {copy_len} indices.")
                # This might indicate issues in EM span mapping or `get_entity_token_indices` if entities are truncated by tokenizer
        
        encoding['entity_token_indices'] = padded_entity_token_indices

        # --- EE Labels (if EEF) ---
        if config.USE_EEF:
            ee_labels = torch.full((self.max_length,), -100, dtype=torch.long) # -100 is CrossEntropyLoss ignore_index
            previous_word_idx = None
            
            # entity_labels and entity_word_indices should be from the (truncated) `entities` list
            for token_idx, current_word_id in enumerate(word_ids):
                if current_word_id is None:
                    ee_labels[token_idx] = -100
                    continue
                
                is_first_token_of_word = (current_word_id != previous_word_idx)
                label_str = "O"

                # Iterate through the (truncated and EM-mapped) entity_word_indices
                for ent_idx in range(len(entity_word_indices)): # Use len(entity_word_indices) which should be consistent
                    ent_start_word_idx, ent_end_word_idx = entity_word_indices[ent_idx]
                    if ent_start_word_idx <= current_word_id <= ent_end_word_idx:
                        # Get base_label corresponding to this entity
                        # entity_labels should have the same length as entities (and entity_word_indices after EM)
                        base_label = entity_labels[ent_idx]
                        if is_first_token_of_word and current_word_id == ent_start_word_idx:
                            label_str = f"B-{base_label}"
                        else:
                            label_str = f"I-{base_label}"
                        break
                
                if is_first_token_of_word:
                    ee_labels[token_idx] = config.LABEL_TO_ID.get(label_str, config.LABEL_TO_ID["O"])
                else:
                    prev_token_label_id = ee_labels[token_idx-1].item() # ee_labels[token_idx-1] is already a tensor
                    if prev_token_label_id != -100:
                        prev_label_str_full = config.ID_TO_LABEL[prev_token_label_id]
                        if prev_label_str_full.startswith("B-"):
                            current_actual_label = f"I-{prev_label_str_full[2:]}"
                            ee_labels[token_idx] = config.LABEL_TO_ID.get(current_actual_label, config.LABEL_TO_ID["O"])
                        elif prev_label_str_full.startswith("I-"):
                            ee_labels[token_idx] = config.LABEL_TO_ID.get(prev_label_str_full, config.LABEL_TO_ID["O"])
                        else: # Was "O"
                            ee_labels[token_idx] = config.LABEL_TO_ID["O"]
                    else: # Previous token was special token
                        ee_labels[token_idx] = config.LABEL_TO_ID["O"] # Should ideally be -100 if prev was special
                previous_word_idx = current_word_id
            encoding['ee_labels'] = ee_labels


        final_encoding = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 else v for k, v in encoding.items()}
        final_encoding['image_size'] = torch.tensor([width, height], dtype=torch.long)
        final_encoding['num_entities'] = num_actual_entities_to_process # Store the *actual* number of entities processed

        return final_encoding

# collate_fn_bbs remains the same
def collate_fn_bbs(batch):
    from torch.utils.data.dataloader import default_collate
    # Filter out None items if any __getitem__ failed and returned None
    # batch = [b for b in batch if b is not None]
    # if not batch:
    #     return None # Or handle as an error / empty batch
    processed_batch = default_collate(batch)

    if config.USE_BBS and len(batch) > 0 and processed_batch['input_ids'].shape[0] > 1:
        batch_size = processed_batch['input_ids'].shape[0]
        perm = torch.randperm(batch_size)
        shuffled_bboxes = processed_batch['bbox'][perm]
        processed_batch['bbox'] = shuffled_bboxes
    return processed_batch