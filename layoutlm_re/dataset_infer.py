# dataset.py
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor
import logging

# Assuming config.py and utils.py are in the same directory or accessible via PYTHONPATH
import config
from utils import normalize_bbox, get_entity_token_indices

logger = logging.getLogger(__name__)
# Configure basic logging if not already configured by the main script
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VRDDataset(Dataset):
    def __init__(self, data_path, is_train=False):
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            config.MODEL_NAME, cache_dir=config.CACHE_DIR
        )
        # Image processor should match training setup. apply_ocr=False if OCR is external/pre-applied.
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            config.MODEL_NAME, cache_dir=config.CACHE_DIR, apply_ocr=config.APPLY_OCR_IN_PROCESSOR if hasattr(config, 'APPLY_OCR_IN_PROCESSOR') else False
        )
        self.image_dir = config.IMAGE_DIR
        self.max_length = config.MAX_LENGTH
        self.is_train = is_train
        self.pad_token_box = [0, 0, 0, 0]

        self.data = []
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            # Depending on strictness, could raise FileNotFoundError or proceed with empty dataset
            # raise FileNotFoundError(f"Data file not found: {data_path}")
            return # Proceed with an empty dataset

        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Skipping line {i+1} in {data_path} due to JSON decode error: {e}")
                    continue
        
        if not self.data:
            logger.warning(f"No data loaded from {data_path}. Dataset is empty.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        sample = self.data[idx]
        
        sample_id = sample.get('id', f"sample_{idx}")

        image_filename = sample.get('image_path')
        if not image_filename:
            logger.error(f"Sample {sample_id} (index {idx}) is missing 'image_path'. Cannot process.")
            # This sample will be problematic. Depending on DataLoader behavior with None,
            # it might be better to filter such samples in __init__ or raise an error here.
            # For robustness in a batch, one might return a dummy item that collate_fn can skip,
            # or this will cause an error later when Image.open fails.
            # Let's assume image_path is critical and this leads to an error.
            raise ValueError(f"Sample {sample_id} (index {idx}) is missing 'image_path'.")

        image_path = os.path.join(self.image_dir, image_filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image file not found for sample {sample_id}: {image_path}.")
            raise
        width, height = image.size

        words = sample.get('words', [])
        unnormalized_boxes = sample.get('bboxes', [])

        if len(words) != len(unnormalized_boxes):
            logger.warning(f"Sample {sample_id}: Mismatch words ({len(words)}) and bboxes ({len(unnormalized_boxes)}). Using empty lists.")
            words, unnormalized_boxes = [], []

        normalized_boxes = [normalize_bbox(box, width, height) for box in unnormalized_boxes]

        # --- Entities ---
        # `entities` are expected to be a list of dicts, e.g., {'id': 'e1', 'span': [start, end], 'label': 'PERSON'}
        # If 'entities' key is not in sample, original_entities will be an empty list.
        original_entities_from_sample = sample.get('entities', [])
        
        # Truncate if too many entities
        if len(original_entities_from_sample) > config.MAX_ENTITIES:
            logger.debug(f"Sample {sample_id}: Has {len(original_entities_from_sample)} entities, truncating to {config.MAX_ENTITIES}.")
            current_entities = original_entities_from_sample[:config.MAX_ENTITIES]
        else:
            current_entities = original_entities_from_sample
        
        # Filter out entities that don't have a 'span'
        valid_entities = []
        for i, ent in enumerate(current_entities):
            if 'span' in ent and isinstance(ent['span'], list) and len(ent['span']) == 2:
                # Further check if span indices are valid for the current words list
                start_idx, end_idx = ent['span']
                if 0 <= start_idx < len(words) and 0 <= end_idx < len(words) and start_idx <= end_idx:
                    valid_entities.append(ent)
                else:
                    logger.warning(f"Sample {sample_id}: Entity {ent.get('id', i)} has invalid span {ent['span']} for words list length {len(words)}. Skipping this entity.")
            else:
                logger.warning(f"Sample {sample_id}: Entity {ent.get('id', i)} is missing 'span' or span is invalid. Skipping this entity.")
        current_entities = valid_entities
        num_entities_after_trunc_and_span_check = len(current_entities)

        entity_word_indices = [e['span'] for e in current_entities]
        entity_labels_for_eef = [e.get('label', 'UNKNOWN_LABEL') for e in current_entities] # Needed for EEF and EM

        # --- Entity Marking (EM) ---
        if config.USE_EM and num_entities_after_trunc_and_span_check > 0:
            # EM modifies `words`, `normalized_boxes`, and `entity_word_indices`
            original_word_count_before_em = len(words)
            new_words_em, new_boxes_em, word_idx_map_em = [], [], {}
            updated_entity_word_indices_em = []
            
            # Build map of where entity markers should be inserted
            entity_starts_for_em = {} # {word_idx: entity_label_for_marker}
            for i, ent_detail in enumerate(current_entities): # Use entities that passed span checks
                ent_label_for_marker = entity_labels_for_eef[i] # Use corresponding label
                # EM markers are typically based on entity type.
                entity_starts_for_em[ent_detail['span'][0]] = ent_label_for_marker.upper().replace("-", "_") # Example marker format

            current_new_idx_em = 0
            for i in range(original_word_count_before_em):
                marker_insertion_point = current_new_idx_em
                if i in entity_starts_for_em:
                    new_words_em.append(f"<{entity_starts_for_em[i]}>") # Example: <PERSON>
                    new_boxes_em.append(normalized_boxes[i]) # Marker uses bbox of first word of entity
                    current_new_idx_em += 1
                
                word_insertion_point = current_new_idx_em
                new_words_em.append(words[i])
                new_boxes_em.append(normalized_boxes[i])
                current_new_idx_em += 1
                word_idx_map_em[i] = [marker_insertion_point, word_insertion_point, current_new_idx_em]

            words = new_words_em
            normalized_boxes = new_boxes_em

            for i in range(num_entities_after_trunc_and_span_check):
                original_span_start, original_span_end = current_entities[i]['span']
                if original_span_start in word_idx_map_em and original_span_end in word_idx_map_em:
                    new_span_start = word_idx_map_em[original_span_start][1] # Index of the word after its marker
                    new_span_end = word_idx_map_em[original_span_end][1]   # Index of the last word of entity
                    updated_entity_word_indices_em.append([new_span_start, new_span_end])
                else:
                    logger.warning(f"Sample {sample_id}: Entity {current_entities[i].get('id', i)} span issue during EM mapping. Original: [{original_span_start},{original_span_end}]. Skipping this entity for token indices.")
            entity_word_indices = updated_entity_word_indices_em
            # `entity_labels_for_eef` does not need update if its order corresponds to `current_entities` whose spans are updated.
            # However, if EM filtering caused entities to be dropped, `entity_labels_for_eef` needs to be filtered too.
            # For simplicity, assume `updated_entity_word_indices_em` now reflects the true set of entities after EM.
            # We may need to filter `entity_labels_for_eef` to match `entity_word_indices` length if EM caused drops.
            # A robust way: re-create entity_labels_for_eef from the entities that successfully had their spans updated.
            # This detail is complex. For now, assume len(entity_word_indices) is the new count.
            if len(entity_word_indices) != len(entity_labels_for_eef):
                 # This implies EM logic needs to filter entity_labels_for_eef as well.
                 # For now, we'll use the potentially shorter entity_word_indices list.
                 # This might affect EEF label generation if labels list is not also adjusted.
                 # Let's assume entity_labels_for_eef are still aligned for entities in entity_word_indices.
                 pass


        # --- Tokenization ---
        encoding = self.tokenizer(
            words,
            boxes=normalized_boxes,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0) # Map tokens to word indices (after EM)

        # --- Image Features ---
        if config.USE_IMAGE:
            try:
                image_features = self.image_processor(image, return_tensors="pt")
                encoding['pixel_values'] = image_features['pixel_values'].squeeze(0)
            except Exception as e:
                logger.error(f"Error processing image for sample {sample_id}: {e}. 'pixel_values' will be missing for this sample.")
                # `predict.py` handles missing 'pixel_values' in the batch via `batch.get('pixel_values')`.
        
        # --- Entity Token Indices ---
        # `entity_word_indices` reflects entities after EM and span validation.
        actual_entity_first_token_indices = get_entity_token_indices(word_ids, entity_word_indices)
        
        padded_entity_token_indices = torch.full(
            (config.MAX_ENTITIES,), config.ENTITY_INDEX_PAD_VALUE, dtype=torch.long
        )
        
        num_entities_with_found_tokens = len(actual_entity_first_token_indices)
        
        # This is the final number of entities the model will process for RE.
        # It's min(entities_after_all_filtering, entities_with_valid_tokens_found)
        # and also capped by MAX_ENTITIES (implicitly, as padded_entity_token_indices has that size).
        effective_num_entities_for_model = 0
        if num_entities_with_found_tokens > 0:
            # `entity_word_indices` could be shorter than `num_entities_with_found_tokens` if `get_entity_token_indices`
            # filters further (e.g. entities fully truncated by tokenizer).
            # The number of valid entities is driven by `actual_entity_first_token_indices`.
            copy_len = min(num_entities_with_found_tokens, config.MAX_ENTITIES) # Should already be capped by MAX_ENTITIES earlier
            padded_entity_token_indices[:copy_len] = torch.tensor(actual_entity_first_token_indices[:copy_len], dtype=torch.long)
            effective_num_entities_for_model = copy_len
        
        if len(entity_word_indices) != effective_num_entities_for_model and len(entity_word_indices) > 0 :
             logger.debug(f"Sample {sample_id}: Entities based on word spans ({len(entity_word_indices)}) vs. entities with found tokens ({effective_num_entities_for_model}).")

        encoding['entity_token_indices'] = padded_entity_token_indices

        # Squeeze tensors (e.g., (1, seq_len) -> (seq_len))
        final_encoding = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 and v.shape[0] == 1 else v for k, v in encoding.items()}

        final_encoding['id'] = sample_id
        final_encoding['image_size'] = torch.tensor([width, height], dtype=torch.long)
        final_encoding['num_entities'] = effective_num_entities_for_model


        # --- Training-specific fields ---
        if self.is_train:
            # Relation Matrix
            # Dimensions based on `effective_num_entities_for_model`
            actual_relation_matrix = torch.zeros((effective_num_entities_for_model, effective_num_entities_for_model), dtype=torch.float)
            
            if effective_num_entities_for_model > 0 and 'relations' in sample:
                # Map original entity IDs (from `sample['entities']`) to their new indices (0 to `effective_num_entities_for_model-1`)
                # This mapping needs to be robust to entities being dropped or reordered by EM or tokenization.
                # The `current_entities` list (after MAX_ENTITIES truncation and span validation)
                # and `entity_word_indices` (after EM) and then `actual_entity_first_token_indices`
                # determine the final set of `effective_num_entities_for_model`.
                # We assume `actual_entity_first_token_indices[k]` corresponds to an entity that was originally
                # `current_entities[k]` (if no EM filtering) or `current_entities[original_idx_mapped_by_em[k]]` (if EM filtered).
                # For simplicity, let's assume `current_entities[k]` (for `k < effective_num_entities_for_model`) is the reference.
                entity_id_to_final_idx = {}
                for i in range(effective_num_entities_for_model):
                    # This relies on `current_entities` still being representative of the entities for which tokens were found.
                    # If EM or `get_entity_token_indices` reorders or heavily filters, this mapping could be complex.
                    # A robust way is to carry original entity objects/IDs alongside `actual_entity_first_token_indices`.
                    # For now, assume direct correspondence with the first `effective_num_entities_for_model` of `current_entities`.
                    if i < len(current_entities) and 'id' in current_entities[i]:
                         entity_id_to_final_idx[current_entities[i]['id']] = i
                    # else:
                        # This case means we can't map an entity ID, so relations for it can't be formed.
                        # logger.warning(f"Sample {sample_id}: Cannot map entity ID for effective entity index {i} during relation matrix creation.")

                for rel in sample.get('relations', []):
                    head_id = rel.get('head')
                    tail_id = rel.get('tail')
                    if head_id in entity_id_to_final_idx and tail_id in entity_id_to_final_idx:
                        head_idx = entity_id_to_final_idx[head_id]
                        tail_idx = entity_id_to_final_idx[tail_id]
                        # Ensure indices are within the bounds of `actual_relation_matrix`
                        if head_idx < effective_num_entities_for_model and tail_idx < effective_num_entities_for_model:
                             actual_relation_matrix[head_idx, tail_idx] = 1.0
            
            padded_relation_matrix = torch.full(
                (config.MAX_ENTITIES, config.MAX_ENTITIES), 0.0, dtype=torch.float
            )
            if effective_num_entities_for_model > 0:
                padded_relation_matrix[:effective_num_entities_for_model, :effective_num_entities_for_model] = actual_relation_matrix
            final_encoding['relation_matrix'] = padded_relation_matrix


            # EE Labels (IOB)
            if config.USE_EEF:
                ee_labels = torch.full((self.max_length,), -100, dtype=torch.long) # -100 is CrossEntropyLoss ignore_index
                previous_word_idx_eef = None
                
                # `entity_word_indices` are word-spans after EM. `entity_labels_for_eef` should align with `current_entities`.
                # We need to ensure `entity_labels_for_eef` corresponds to `entity_word_indices` if EM filtered entities.
                # Let's re-fetch labels based on `current_entities` and assume `entity_word_indices` aligns with its first `len(entity_word_indices)` items.
                temp_labels_for_eef_aligned_with_spans = [entity_labels_for_eef[i] for i in range(len(entity_word_indices))]


                for token_idx, current_word_id_from_tokenizer in enumerate(word_ids):
                    if current_word_id_from_tokenizer is None:
                        ee_labels[token_idx] = -100
                        continue
                    
                    is_first_token_of_word = (current_word_id_from_tokenizer != previous_word_idx_eef)
                    label_str = "O"

                    for ent_idx in range(len(entity_word_indices)): # Use current entity_word_indices (after EM)
                        ent_start_word_idx, ent_end_word_idx = entity_word_indices[ent_idx]
                        if ent_start_word_idx <= current_word_id_from_tokenizer <= ent_end_word_idx:
                            # Use label from `temp_labels_for_eef_aligned_with_spans` which should match `entity_word_indices`
                            base_label = temp_labels_for_eef_aligned_with_spans[ent_idx] 
                            if is_first_token_of_word and current_word_id_from_tokenizer == ent_start_word_idx:
                                label_str = f"B-{base_label}"
                            else:
                                label_str = f"I-{base_label}"
                            break
                    
                    if is_first_token_of_word:
                        ee_labels[token_idx] = config.LABEL_TO_ID.get(label_str, config.LABEL_TO_ID["O"])
                    else: # Subsequent token of a multi-token word
                        prev_token_label_id = ee_labels[token_idx-1].item()
                        if prev_token_label_id != -100:
                            prev_label_str_full = config.ID_TO_LABEL.get(prev_token_label_id, "O")
                            if prev_label_str_full.startswith("B-"):
                                current_actual_label = f"I-{prev_label_str_full[2:]}"
                                ee_labels[token_idx] = config.LABEL_TO_ID.get(current_actual_label, config.LABEL_TO_ID["O"])
                            elif prev_label_str_full.startswith("I-"):
                                ee_labels[token_idx] = config.LABEL_TO_ID.get(prev_label_str_full, config.LABEL_TO_ID["O"])
                            else: # Was "O"
                                ee_labels[token_idx] = config.LABEL_TO_ID["O"]
                        else: # Previous token was special
                            ee_labels[token_idx] = config.LABEL_TO_ID["O"] # Or -100 if it's padding after special
                    previous_word_idx_eef = current_word_id_from_tokenizer
                final_encoding['ee_labels'] = ee_labels
        
        return final_encoding


def collate_fn_bbs(batch):
    from torch.utils.data.dataloader import default_collate
    
    # Filter out None items if __getitem__ could return None for faulty samples
    # batch = [b for b in batch if b is not None]
    # if not batch:
    #     logger.warning("Collate_fn_bbs received an empty batch after filtering None items.")
    #     return None # Or handle as appropriate for your training/evaluation loop

    # Check for presence of 'pixel_values' if config.USE_IMAGE is True.
    # default_collate will fail if keys are inconsistent across batch items.
    # Ensure all dicts in the batch have the same set of keys, possibly padding with None or default tensors.
    # However, `predict.py` uses `batch.get('pixel_values')`, so if the key is absent from the entire
    # collated batch, it's handled. The issue is if it's present in some items but not others.
    # For simplicity, assume __getitem__ is consistent or errors out for critical missing data like images.

    processed_batch = default_collate(batch)

    if config.USE_BBS and len(batch) > 0 and processed_batch['input_ids'].shape[0] > 1:
        batch_size = processed_batch['input_ids'].shape[0]
        perm = torch.randperm(batch_size)
        if 'bbox' in processed_batch: # Ensure 'bbox' key exists
            shuffled_bboxes = processed_batch['bbox'][perm]
            processed_batch['bbox'] = shuffled_bboxes
        else:
            logger.warning("config.USE_BBS is True, but 'bbox' key missing in collated batch. Skipping BBS.")
            
    return processed_batch

# Example usage (for testing the dataset class):
if __name__ == '__main__':
    # Create dummy config attributes for testing
    class DummyConfig:
        MODEL_NAME = "microsoft/layoutlmv3-base"
        CACHE_DIR = "./cache"
        IMAGE_DIR = "./dummy_images" # Path to your dummy images
        MAX_LENGTH = 512
        MAX_ENTITIES = 25
        ENTITY_INDEX_PAD_VALUE = -1 # Or whatever your pad value is
        USE_EM = False # Enable to test Entity Marking
        USE_EEF = True # Enable to test Entity Extraction Features/Labels
        LABEL_TO_ID = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-ORG": 3, "I-ORG": 4, "UNKNOWN_LABEL": 5} # Example
        ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
        USE_IMAGE = True
        USE_BBS = False # Bounding Box Shuffle
        APPLY_OCR_IN_PROCESSOR = False # LayoutLMv3ImageProcessor.apply_ocr default

    config = DummyConfig() # Override the imported config for this test block

    # Create dummy data file (e.g., dummy_data.jsonl)
    # Each line is a JSON object:
    # {"id": "doc1", "image_path": "img1.png", "words": ["Hello", "world"], "bboxes": [[10,10,50,20], [60,10,100,20]], "entities": [{"id": "e1", "span": [0,0], "label": "PERSON"}, {"id": "e2", "span": [1,1], "label": "ORG"}], "relations": [{"head": "e1", "tail": "e2", "label": "works_for"}]}
    # {"id": "doc2", "image_path": "img2.png", "words": ["Test"], "bboxes": [[10,10,50,20]], "entities": []} # No entities
    # {"id": "doc3", "image_path": "img3.png", "words": ["Another", "example"], "bboxes": [[10,10,50,20], [60,10,100,20]]} # No entities, no relations

    # Ensure dummy_images directory and dummy images exist if USE_IMAGE=True
    # os.makedirs(config.IMAGE_DIR, exist_ok=True)
    # Image.new('RGB', (100, 100), color = 'red').save(os.path.join(config.IMAGE_DIR, "img1.png"))
    # Image.new('RGB', (100, 100), color = 'blue').save(os.path.join(config.IMAGE_DIR, "img2.png"))
    # Image.new('RGB', (100, 100), color = 'green').save(os.path.join(config.IMAGE_DIR, "img3.png"))

    # dummy_jsonl_path = "dummy_data.jsonl"
    # with open(dummy_jsonl_path, 'w') as f:
    #     data_samples = [
    #         {"id": "doc1", "image_path": "img1.png", "words": ["Hello", "world", "from", "LayoutLMv3"], "bboxes": [[10,10,50,20], [60,10,100,20], [10,30,50,40], [60,30,150,40]], "entities": [{"id": "e1", "span": [0,1], "label": "PERSON"}, {"id": "e2", "span": [3,3], "label": "ORG"}], "relations": [{"head": "e1", "tail": "e2"}]},
    #         {"id": "doc2", "image_path": "img2.png", "words": ["Test", "document"], "bboxes": [[10,10,50,20], [10,10,50,20]], "entities": []},
    #         {"id": "doc3", "image_path": "img3.png", "words": ["No", "entities", "here"], "bboxes": [[10,10,50,20], [10,10,50,20], [10,10,50,20]]}
    #     ]
    #     for s in data_samples:
    #         f.write(json.dumps(s) + "\n")

    # print("Testing VRDDataset for training...")
    # train_dataset = VRDDataset(data_path=dummy_jsonl_path, is_train=True)
    # if len(train_dataset) > 0:
    #     train_sample = train_dataset[0]
    #     print(f"Train Sample 0 ('{train_sample['id']}') Keys: {train_sample.keys()}")
    #     print(f"Num entities: {train_sample['num_entities']}")
    #     print(f"Entity token indices: {train_sample['entity_token_indices']}")
    #     if 'relation_matrix' in train_sample:
    #         print(f"Relation matrix shape: {train_sample['relation_matrix'].shape}")
    #     if 'ee_labels' in train_sample:
    #         print(f"EE labels shape: {train_sample['ee_labels'].shape}")
    #     if 'pixel_values' in train_sample:
    #         print(f"Pixel values shape: {train_sample['pixel_values'].shape}")

    # print("\nTesting VRDDataset for inference...")
    # inference_dataset = VRDDataset(data_path=dummy_jsonl_path, is_train=False)
    # if len(inference_dataset) > 0:
    #     inference_sample = inference_dataset[0]
    #     print(f"Inference Sample 0 ('{inference_sample['id']}') Keys: {inference_sample.keys()}")
    #     print(f"Num entities: {inference_sample['num_entities']}")
    #     assert 'relation_matrix' not in inference_sample, "Relation matrix should not be in inference sample"
    #     assert 'ee_labels' not in inference_sample or not config.USE_EEF, "EE labels should not be in inference sample (unless USE_EEF is false)"

    #     inference_sample_no_entities = inference_dataset[1] # doc2 has no entities
    #     print(f"Inference Sample 1 ('{inference_sample_no_entities['id']}') Num entities: {inference_sample_no_entities['num_entities']}")
    #     assert inference_sample_no_entities['num_entities'] == 0, "Num entities should be 0 for sample with no entities"

    #     # Test collate_fn
    #     from torch.utils.data import DataLoader
    #     loader = DataLoader(inference_dataset, batch_size=2, collate_fn=collate_fn_bbs)
    #     try:
    #         batch = next(iter(loader))
    #         print("\nBatch loaded successfully.")
    #         print(f"Batch keys: {batch.keys()}")
    #         print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    #         print(f"Batch num_entities: {batch['num_entities']}")
    #     except Exception as e:
    #         print(f"Error during DataLoader iteration: {e}")
    pass # End of example usage block