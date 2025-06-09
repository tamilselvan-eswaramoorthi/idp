import json
import os
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

def validate_jsonl_file(jsonl_path):
    """
    Validates that a JSONL file has the correct structure for the VRDDataset.
    
    Args:
        jsonl_path: Path to the JSONL file
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print(f"Validating {jsonl_path}...")
    
    required_fields = ['words', 'bboxes', 'entities']
    optional_fields = ['image_path', 'relations', 'id']
    
    line_num = 0
    errors = []
    warnings = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Check required fields
                    for field in required_fields:
                        if field not in data:
                            errors.append(f"Line {line_num}: Missing required field '{field}'")
                            continue
                    
                    # Check words and bboxes have same length
                    if 'words' in data and 'bboxes' in data:
                        if len(data['words']) != len(data['bboxes']):
                            errors.append(f"Line {line_num}: Number of words ({len(data['words'])}) doesn't match number of bboxes ({len(data['bboxes'])})")
                    
                    # Check bbox format
                    if 'bboxes' in data:
                        for i, bbox in enumerate(data['bboxes']):
                            if not isinstance(bbox, list) or len(bbox) != 4:
                                errors.append(f"Line {line_num}: Invalid bbox format at index {i}: {bbox}")
                    
                    # Check entities structure
                    if 'entities' in data:
                        for i, entity in enumerate(data['entities']):
                            if 'id' not in entity:
                                errors.append(f"Line {line_num}: Entity at index {i} missing 'id' field")
                            if 'span' not in entity:
                                errors.append(f"Line {line_num}: Entity at index {i} missing 'span' field")
                            elif not isinstance(entity['span'], list) or len(entity['span']) != 2:
                                errors.append(f"Line {line_num}: Entity at index {i} has invalid 'span' format: {entity['span']}")
                            elif 'words' in data:
                                if entity['span'][0] < 0 or entity['span'][1] >= len(data['words']):
                                    errors.append(f"Line {line_num}: Entity at index {i} has out-of-bounds span: {entity['span']} (words length: {len(data['words'])})")
                    
                    # Check relations structure
                    if 'relations' in data:
                        for i, relation in enumerate(data['relations']):
                            if 'head' not in relation:
                                errors.append(f"Line {line_num}: Relation at index {i} missing 'head' field")
                            if 'tail' not in relation:
                                errors.append(f"Line {line_num}: Relation at index {i} missing 'tail' field")
                            if 'label' not in relation:
                                errors.append(f"Line {line_num}: Relation at index {i} missing 'label' field")
                            
                            if 'head' in relation and 'tail' in relation and 'entities' in data:
                                if relation['head'] == relation['tail']:
                                    warnings.append(f"Line {line_num}: Relation at index {i} has same head and tail: {relation['head']}")
                                
                                entity_ids = [e['id'] for e in data['entities']]
                                if relation['head'] not in entity_ids:
                                    errors.append(f"Line {line_num}: Relation at index {i} has invalid head id: {relation['head']}")
                                if relation['tail'] not in entity_ids:
                                    errors.append(f"Line {line_num}: Relation at index {i} has invalid tail id: {relation['tail']}")
                
                except json.JSONDecodeError:
                    errors.append(f"Line {line_num}: Invalid JSON")
                except Exception as e:
                    errors.append(f"Line {line_num}: Error processing line: {str(e)}")
    
    except Exception as e:
        errors.append(f"Error opening file: {str(e)}")
    
    # Print validation results
    if errors:
        print(f"\nFound {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\nFound {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print(f"\nValidation passed! File contains {line_num} valid samples.")
        return True
    elif not errors:
        print(f"\nValidation passed with {len(warnings)} warnings. File contains {line_num} samples.")
        return True
    else:
        print(f"\nValidation failed with {len(errors)} errors.")
        return False

def normalize_text(text):
    """
    Normalize text for more consistent processing.
    
    Args:
        text (str): Input text string
        
    Returns:
        str: Normalized text
    """
    # Replace multiple spaces, tabs, newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing spaces
    text = text.strip()
    return text

def convert_funsd_to_jsonl(funsd_json_path, image_path_prefix=None):
    """
    Converts a single FUNSD JSON annotation file to a JSONL-compatible dictionary.

    Args:
        funsd_json_path (str): The path to the FUNSD JSON file.
        image_path_prefix (str, optional): Prefix for image path. If None, just use filename.

    Returns:
        dict: A dictionary structured for the VRDDataset.
    """
    with open(funsd_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc_id = Path(funsd_json_path).stem
    image_filename = f"{doc_id}.png"
    
    all_words = []
    all_bboxes = []
    all_ner_tags = []
    word_counter = 0
    entity_data = []
    funsd_id_to_entity_data = {}
    
    # First pass: Process all annotations to gather words, bboxes, and create entities
    for ann_idx, annotation in enumerate(data['form']):
        if not all(k in annotation for k in ['id', 'text', 'box', 'label', 'words']):
            print(f"Warning: Skipping malformed annotation (missing keys) in {funsd_json_path}, index {ann_idx}")
            continue
            
        entity_label = annotation['label'].upper() if annotation.get('label') else "OTHER"
        original_funsd_id = annotation['id']
        
        # Skip 'other' labels if we want to follow converter.py's approach
        # if entity_label == "OTHER":
        #    continue
        
        # Process words with text normalization
        valid_words = []
        valid_boxes = []
        
        if annotation['words']:
            for word_info in annotation['words']:
                if not all(k in word_info for k in ['text', 'box']):
                    continue
                    
                normalized_text = normalize_text(word_info['text'])
                if normalized_text:  # Skip empty strings after normalization
                    valid_words.append(normalized_text)
                    # Ensure bounding box coordinates are integers
                    valid_boxes.append([int(coord) for coord in word_info['box']])
        
        # Skip entities with no valid words
        if not valid_words:
            continue
            
        entity_start_word_idx = word_counter
        
        # Add words to the global list
        for i, (word, box) in enumerate(zip(valid_words, valid_boxes)):
            all_words.append(word)
            all_bboxes.append(box)
            
            # Add BIO tags for NER (from create_dataset.py)
            bio_label = entity_label
            if i == 0: 
                all_ner_tags.append(f"B-{bio_label}")
            else: 
                all_ner_tags.append(f"I-{bio_label}")
                
            word_counter += 1
            
        entity_end_word_idx = word_counter - 1
        
        # Store entity data with position information for sorting
        y_pos = annotation['box'][1] if annotation.get('box') else valid_boxes[0][1]
        x_pos = annotation['box'][0] if annotation.get('box') else valid_boxes[0][0]
        
        entity_info = {
            "original_funsd_id": original_funsd_id,
            "text": normalize_text(annotation['text']),
            "label": entity_label,
            "span": [entity_start_word_idx, entity_end_word_idx],
            "box": annotation['box'],
            "position": (y_pos, x_pos)  # For sorting by position
        }
        
        entity_data.append(entity_info)
        funsd_id_to_entity_data[original_funsd_id] = entity_info

    # Sort entities by position (top to bottom, left to right)
    sorted_entities = sorted(entity_data, key=lambda x: x["position"])
    
    # Create final entities with sequential IDs
    final_entities = []
    funsd_id_to_sequential_id = {}
    
    for idx, entity_info in enumerate(sorted_entities):
        funsd_id_to_sequential_id[entity_info["original_funsd_id"]] = idx
        
        final_entities.append({
            "id": idx,
            "original_funsd_id": entity_info["original_funsd_id"],
            "label": entity_info["label"],
            "span": entity_info["span"], 
            "text": entity_info["text"],
            "box": entity_info["box"]
        })

    # Process relations
    relations = []
    relation_pairs = set()  # To track unique pairs (optional)
    
    for annotation in data['form']:
        if 'id' not in annotation or 'linking' not in annotation:
            continue
            
        current_funsd_id = annotation['id']
        if current_funsd_id not in funsd_id_to_sequential_id:
            continue
            
        for link_pair in annotation['linking']:
            if len(link_pair) != 2:
                print(f"Warning: Malformed link pair {link_pair} in {funsd_json_path} for annotation ID {current_funsd_id}. Skipping.")
                continue
                
            funsd_head_id, funsd_tail_id = link_pair
            
            if funsd_head_id not in funsd_id_to_sequential_id or funsd_tail_id not in funsd_id_to_sequential_id:
                continue
                
            head_id = funsd_id_to_sequential_id[funsd_head_id]
            tail_id = funsd_id_to_sequential_id[funsd_tail_id]
            
            # Optionally check for and skip duplicate relations
            relation_key = (head_id, tail_id)
            if relation_key in relation_pairs:
                continue
                
            relation_pairs.add(relation_key)
            
            relations.append({
                "head": head_id,
                "tail": tail_id,
                "label": "LINK"
            })
    
    # Create image path
    if image_path_prefix:
        image_path = str(Path(image_path_prefix) / image_filename)
    else:
        image_path = image_filename
        
    output_dict = {
        "id": doc_id,
        "image_path": image_path,
        "words": all_words,
        "bboxes": all_bboxes,
        "ner_tags": all_ner_tags,
        "entities": final_entities,
        "relations": relations
    }
    
    return output_dict

def analyze_jsonl_data(jsonl_path):
    """
    Analyzes a JSONL file containing document relation data.
    
    Args:
        jsonl_path: Path to the JSONL file
    """
    print(f"Analyzing data in {jsonl_path}")
    
    # Stats to collect
    num_samples = 0
    num_words_list = []
    num_entities_list = []
    num_relations_list = []
    entity_labels = Counter()
    relation_labels = Counter()
    
    # Load and analyze each sample
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                num_samples += 1
                
                # Word stats
                words = sample.get('words', [])
                num_words_list.append(len(words))
                
                # Entity stats
                entities = sample.get('entities', [])
                num_entities_list.append(len(entities))
                
                # Collect entity labels if available
                for entity in entities:
                    if 'label' in entity:
                        entity_labels[entity['label']] += 1
                
                # Relation stats
                relations = sample.get('relations', [])
                num_relations_list.append(len(relations))
                
                # Collect relation labels
                for relation in relations:
                    if 'label' in relation:
                        relation_labels[relation['label']] += 1
            
            except json.JSONDecodeError:
                print(f"Error: Could not parse a line in {jsonl_path}")
            except Exception as e:
                print(f"Error processing line: {e}")
    
    # Calculate statistics
    print(f"\nTotal samples: {num_samples}")
    
    if num_words_list:
        print(f"\nWord statistics:")
        print(f"  Min words per sample: {min(num_words_list)}")
        print(f"  Max words per sample: {max(num_words_list)}")
        print(f"  Avg words per sample: {sum(num_words_list) / len(num_words_list):.2f}")
    
    if num_entities_list:
        print(f"\nEntity statistics:")
        print(f"  Min entities per sample: {min(num_entities_list)}")
        print(f"  Max entities per sample: {max(num_entities_list)}")
        print(f"  Avg entities per sample: {sum(num_entities_list) / len(num_entities_list):.2f}")
        print(f"  Total entities: {sum(num_entities_list)}")
        
        print("\nEntity label distribution:")
        for label, count in entity_labels.most_common():
            print(f"  {label}: {count} ({count / sum(entity_labels.values()) * 100:.1f}%)")
    
    if num_relations_list:
        print(f"\nRelation statistics:")
        print(f"  Min relations per sample: {min(num_relations_list)}")
        print(f"  Max relations per sample: {max(num_relations_list)}")
        print(f"  Avg relations per sample: {sum(num_relations_list) / len(num_relations_list):.2f}")
        print(f"  Total relations: {sum(num_relations_list)}")
        
        print("\nRelation label distribution:")
        for label, count in relation_labels.most_common():
            print(f"  {label}: {count} ({count / sum(relation_labels.values()) * 100:.1f}%)")
    
    # Plot histograms if there's data
    if num_entities_list and num_relations_list:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(num_words_list, bins=20)
        plt.title('Words per Sample')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.hist(num_entities_list, bins=max(10, max(num_entities_list)))
        plt.title('Entities per Sample')
        plt.xlabel('Number of Entities')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        plt.hist(num_relations_list, bins=max(10, max(num_relations_list)))
        plt.title('Relations per Sample')
        plt.xlabel('Number of Relations')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        if 'train' in jsonl_path:
            plot_name = 'train_stats.png'
        elif 'test' in jsonl_path:
            plot_name = 'test_stats.png'
        plot_path = os.path.join('plots', plot_name)
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")
        plt.close()
        
    return {
        'num_samples': num_samples,
        'num_entities': sum(num_entities_list),
        'num_relations': sum(num_relations_list),
        'entity_labels': entity_labels,
        'relation_labels': relation_labels
    }


def process_funsd_split(funsd_root_dir, split_name, output_dir, include_image_prefix=True):
    """
    Processes a split of the FUNSD dataset.

    Args:
        funsd_root_dir (str): Path to the root directory of FUNSD.
        split_name (str): "training_data" or "testing_data".
        output_dir (str): Directory to save output JSONL.
        include_image_prefix (bool): Whether to include split name in image paths.
    """
    annotations_dir = os.path.join(funsd_root_dir, split_name, "annotations")
    
    if not os.path.isdir(annotations_dir):
        print(f"Error: Annotation directory not found: {annotations_dir}")
        return
        
    output_filename = "train.jsonl" if "train" in split_name.lower() else "test.jsonl"
    output_file_path = os.path.join(output_dir, output_filename)
    
    # Create image path prefix based on configuration
    image_path_prefix = f"{split_name}/images" if include_image_prefix else None
    
    json_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"Warning: No JSON files found in {annotations_dir}")
        return
        
    print(f"Processing {len(json_files)} files from {split_name}...")
    
    successful_conversions = 0
    total_entities = 0
    total_relations = 0
    
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for filename in tqdm(json_files, desc=f"Converting {split_name}"):
            json_path = os.path.join(annotations_dir, filename)
            try:
                converted_data = convert_funsd_to_jsonl(json_path, image_path_prefix)
                if converted_data:
                    outfile.write(json.dumps(converted_data) + '\n')
                    successful_conversions += 1
                    total_entities += len(converted_data.get('entities', []))
                    total_relations += len(converted_data.get('relations', []))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"Successfully processed {successful_conversions} out of {len(json_files)} files from {split_name}")
    print(f"Total entities: {total_entities}, total relations: {total_relations}")
    print(f"Output saved to {output_file_path}")
    
    if successful_conversions > 0:
        validate_jsonl_file(output_file_path)
    
    # Analyze the output JSONL file
    analyze_jsonl_data(output_file_path)
    return output_file_path

def main():
    # Set up paths
    funsd_root_dir = 'funsd_dataset'  # Update with your actual path
    output_dir = 'data'  # Update with your desired output directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process training data
    print("\n" + "="*50)
    print("Processing FUNSD training data...")
    train_output = process_funsd_split(funsd_root_dir, "training_data", output_dir)
    
    # Process testing data
    print("\n" + "="*50)
    print("Processing FUNSD testing data...")
    test_output = process_funsd_split(funsd_root_dir, "testing_data", output_dir)
    
    print("\n" + "="*50)
    print("Dataset conversion complete!")
    print(f"Training data: {train_output}")
    print(f"Testing data: {test_output}")
    print("="*50)

if __name__ == "__main__":
    main()