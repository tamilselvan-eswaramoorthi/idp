import json
import os
import argparse
from pathlib import Path

def parse_funsd_file(json_path, image_path_prefix):
    """
    Parses a single FUNSD JSON annotation file and converts it into the JSONL format
    expected by the provided VRDDataset class.

    Args:
        json_path (str or Path): Path to the FUNSD JSON annotation file.
        image_path_prefix (str): Prefix for the image path (e.g., "training_data/images").

    Returns:
        dict: A dictionary representing the processed document.
              Returns None if the file is invalid or cannot be processed.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"Warning: Error reading {json_path}: {e}")
        return None

    doc_id = Path(json_path).stem
    image_filename = f"{doc_id}.png"

    if 'form' not in data or not isinstance(data['form'], list):
        print(f"Warning: 'form' key missing or not a list in {json_path}")
        return None

    doc_words = []
    doc_bboxes = []
    doc_ner_tags = [] # Kept for general utility, even if not directly used by VRDDataset's __getitem__
    
    entities = []
    funsd_id_to_entity_sequential_id = {} 
    
    current_word_idx_offset = 0 # Tracks the starting index for words of the current annotation

    # First pass: Process all annotations to gather words, bboxes, and create entities
    for ann_idx, annotation in enumerate(data['form']):
        if not all(k in annotation for k in ['id', 'text', 'box', 'label', 'words']):
            print(f"Warning: Skipping malformed annotation (missing keys) in {json_path}, annotation index {ann_idx}")
            continue

        entity_text = annotation['text']
        entity_box = annotation['box'] 
        entity_label = annotation['label'].upper() if annotation.get('label') else "OTHER"
        original_funsd_id = annotation['id']

        entity_sequential_id = len(entities) # This will be the 'id' for the entity list
        funsd_id_to_entity_sequential_id[original_funsd_id] = entity_sequential_id

        entity_start_word_idx = current_word_idx_offset
        
        num_words_in_entity = 0
        if annotation['words']:
            for i, word_info in enumerate(annotation['words']):
                if not all(k in word_info for k in ['text', 'box']):
                    print(f"Warning: Skipping malformed word (missing keys) in {json_path}, annotation ID {original_funsd_id}, word index {i}")
                    continue
                
                doc_words.append(word_info['text'])
                doc_bboxes.append(word_info['box'])
                
                # Assign NER tags in BIO format
                bio_label = entity_label if entity_label else "OTHER"
                if i == 0: 
                    doc_ner_tags.append(f"B-{bio_label}")
                else: 
                    doc_ner_tags.append(f"I-{bio_label}")
                
                current_word_idx_offset += 1
                num_words_in_entity += 1
        else: # Annotation has no 'words'
            if entity_text.strip():
                 print(f"Warning: Annotation ID {original_funsd_id} in {json_path} has text '{entity_text}' but no detailed 'words'. Entity will have an empty/invalid word span.")
            # No words added to doc_words for this entity, num_words_in_entity remains 0
        
        # Calculate inclusive end word index for the span
        # If num_words_in_entity is 0, entity_end_word_idx will be entity_start_word_idx - 1
        # This [start, start-1] span indicates an entity with no words in doc_words.
        # Your VRDDataset's get_entity_token_indices and EEF logic should handle this.
        entity_end_word_idx = entity_start_word_idx + num_words_in_entity - 1

        entities.append({
            "id": entity_sequential_id,             # Sequential ID for VRDDataset
            "original_funsd_id": original_funsd_id, # Original ID from FUNSD for reference
            "text": entity_text,                    # Full text of the entity
            "box": entity_box,                      # Bounding box of the entity segment
            "label": entity_label,                  # Entity type (e.g., QUESTION, ANSWER)
            "span": [entity_start_word_idx, entity_end_word_idx] # [start_word_idx, end_word_idx] (inclusive)
        })

    # Second pass: Create relations
    relations = []
    for annotation in data['form']:
        if 'id' not in annotation or 'linking' not in annotation:
            continue

        current_funsd_id = annotation['id']
        if current_funsd_id not in funsd_id_to_entity_sequential_id:
            # This can happen if the entity itself was malformed and skipped in the first pass
            # print(f"Warning: Source entity FUNSD ID {current_funsd_id} for linking not found in processed entities for {json_path}. Skipping its links.")
            continue
        
        head_entity_sequential_id = funsd_id_to_entity_sequential_id[current_funsd_id]
        
        for link_pair in annotation['linking']:
            if len(link_pair) != 2:
                print(f"Warning: Malformed link pair {link_pair} in {json_path} for annotation ID {current_funsd_id}. Skipping.")
                continue

            # In FUNSD, the 'linking' field contains pairs [from_id, to_id].
            # 'current_funsd_id' is typically 'from_id'.
            funsd_head_id = link_pair[0]
            funsd_tail_id = link_pair[1]

            if funsd_head_id != current_funsd_id:
                 print(f"Warning: Link source ID {funsd_head_id} in linking pair does not match current annotation ID {current_funsd_id} in {json_path}. Using link pair IDs: {funsd_head_id} -> {funsd_tail_id}.")


            if funsd_head_id not in funsd_id_to_entity_sequential_id:
                # print(f"Warning: Head FUNSD ID {funsd_head_id} of a link not found in processed entities for {json_path}. Skipping link.")
                continue
            if funsd_tail_id not in funsd_id_to_entity_sequential_id:
                # print(f"Warning: Tail FUNSD ID {funsd_tail_id} of a link not found in processed entities for {json_path}. Skipping link.")
                continue
                
            # Use the mapped sequential IDs for the relation
            head_id_for_relation = funsd_id_to_entity_sequential_id[funsd_head_id]
            tail_id_for_relation = funsd_id_to_entity_sequential_id[funsd_tail_id]
            
            # The VRDDataset doesn't seem to use relation labels, but we can include one.
            # The paper implies M_ij=1 means 'i' is key for 'j'.
            # In FUNSD, a link means 'from_id' is linked to 'to_id'. Often 'question' -> 'answer'.
            # So, from_id is head, to_id is tail.
            relation_label = "LINK" 
            
            relations.append({
                "head": head_id_for_relation, 
                "tail": tail_id_for_relation, 
                "label": relation_label # Kept for completeness
            })
            
    processed_doc = {
        "id": doc_id, # Document identifier
        "image_path": str(Path(image_path_prefix) / image_filename), # Relative image path
        "words": doc_words, # List of all words in the document
        "bboxes": doc_bboxes, # List of [x_min, y_min, x_max, y_max] for each word
        "ner_tags": doc_ner_tags, # BIO tags for each word (for general use)
        "entities": entities, # List of entity dicts, each with "id", "label", "span"
        "relations": relations # List of relation dicts, with "head", "tail" using entity "id"
    }
    
    return processed_doc


def process_funsd_split(funsd_root_path, split_name, output_dir):
    """
    Processes a split of the FUNSD dataset (e.g., training or testing).

    Args:
        funsd_root_path (Path): Path to the root of the FUNSD dataset (e.g., ".../funsd_dataset/dataset").
        split_name (str): The name of the split directory, e.g., "training_data" or "testing_data".
        output_dir (Path): Path to the directory where the output .jsonl file will be saved.
    """
    annotations_input_dir = funsd_root_path / split_name / "annotations"
    # The image_path in the output JSONL will be relative to the split's image folder
    # e.g., "training_data/images/doc.png" or "testing_data/images/doc.png"
    # The VRDDataset will then join this with its self.image_dir (which should be funsd_root_path)
    image_path_prefix_for_json = f"{split_name}/images" 

    if not annotations_input_dir.is_dir():
        print(f"Error: Annotation directory not found: {annotations_input_dir}")
        return

    output_filename = "train.jsonl" if "train" in split_name else "test.jsonl"
    output_file_path = output_dir / output_filename

    processed_count = 0
    skipped_count = 0
    json_files = sorted(list(annotations_input_dir.glob("*.json")))
    
    if not json_files:
        print(f"Warning: No JSON files found in {annotations_input_dir}")
        return

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for json_file in json_files:
            processed_document = parse_funsd_file(json_file, image_path_prefix_for_json)
            if processed_document:
                outfile.write(json.dumps(processed_document) + '\n')
                processed_count +=1
            else:
                skipped_count += 1
    
    print(f"Successfully processed {processed_count} documents from {split_name} and saved to {output_file_path}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files in {split_name} due to errors or malformation.")


def main():
    funsd_path = '/home/tamilselvan/Desktop/idp/dataset/FUNSD'
    output_path = 'data'


    funsd_root_path = Path(funsd_path)
    output_dir_path = Path(output_path)

    if not funsd_root_path.is_dir():
        print(f"Error: FUNSD dataset path not found or not a directory: {funsd_root_path}")
        return

    if not (funsd_root_path / "training_data").is_dir() or not (funsd_root_path / "testing_data").is_dir():
        print(f"Error: 'training_data' or 'testing_data' subdirectories not found in {funsd_root_path}. Please provide the path to the 'dataset' directory of FUNSD.")
        return

    output_dir_path.mkdir(parents=True, exist_ok=True)

    print("Processing training data...")
    process_funsd_split(funsd_root_path, "training_data", output_dir_path)
    
    print("\nProcessing testing data...")
    process_funsd_split(funsd_root_path, "testing_data", output_dir_path)
    
    print("\nDataset generation complete.")
    print(f"Output files are located in: {output_dir_path}")
    print("Each entity in the 'entities' list includes a 'span': [start_word_idx, end_word_idx] (inclusive).")

if __name__ == "__main__":
    main()
