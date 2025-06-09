import json
import os, re
from tqdm import tqdm

import config

def validate_jsonl_file(jsonl_path):
    """
    Validates that a JSONL file has the correct structure for the VRDDataset.
    
    Args:
        jsonl_path: Path to the JSONL file
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    
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
    
def convert_funsd_to_jsonl(funsd_json_path):
    """
    Converts a single FUNSD JSON annotation file to a JSONL-compatible dictionary.
    Improved with better text normalization and entity handling.

    Args:
        funsd_json_path (str): The path to the FUNSD JSON file.

    Returns:
        dict: A dictionary structured for the VRDDataset.
    """
    with open(funsd_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_words = []
    all_bboxes = []
    entity_id_map = {}
    relation_pairs = set()
    word_counter = 0
    
    # Improved text normalization
    def normalize_text(text):
        """Normalize text for more consistent processing"""
        # Replace multiple spaces, tabs, newlines with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        text = text.strip()
        return text

    # Process blocks with better handling of empty/invalid cases
    for block in data['form']:
        if not block.get('words') or not (block['words'] and block['words'][0].get('text')):
            continue

        # Verify bounding boxes are valid
        valid_words = []
        for word_info in block['words']:
            if not (isinstance(word_info.get('box'), list) and len(word_info.get('box')) == 4):
                continue
                
            # Normalize text to handle OCR inconsistencies
            word_info['text'] = normalize_text(word_info['text'])
            if word_info['text']:  # Skip empty strings after normalization
                valid_words.append(word_info)
                
        if not valid_words:
            continue
                
        start_word_idx = word_counter
        for word_info in valid_words:
            all_words.append(word_info['text'])
            # Ensure bounding box coordinates are integers
            all_bboxes.append([int(coord) for coord in word_info['box']])
            word_counter += 1
        end_word_idx = word_counter - 1

        if block['label'] != 'other':
            entity_id_map[block['id']] = {
                'id': block['id'],
                'label': block['label'],
                'span': [start_word_idx, end_word_idx]
            }

        for head, tail in block.get('linking', []):
            # Sort to ensure consistent handling regardless of order
            relation_pairs.add(tuple(sorted((head, tail))))

    final_entities = []
    old_id_to_new_idx = {}
    
    # Sort entities by position (top to bottom, left to right)
    entity_list = sorted(entity_id_map.values(), key=lambda x: (
        all_bboxes[x['span'][0]][1],  # y coordinate of first word
        all_bboxes[x['span'][0]][0]   # x coordinate of first word
    ))
    
    for i, entity_info in enumerate(entity_list):
        old_id_to_new_idx[entity_info['id']] = i
        final_entities.append({
            'id': i,
            'span': entity_info['span'],
            'label': entity_info['label']
        })

    final_relations = []
    for head_id, tail_id in relation_pairs:
        if head_id in old_id_to_new_idx and tail_id in old_id_to_new_idx:
            new_head_idx = old_id_to_new_idx[head_id]
            new_tail_idx = old_id_to_new_idx[tail_id]
            final_relations.append({
                'head': new_head_idx,
                'tail': new_tail_idx,
                'label': 'LINK'
            })

    image_filename = os.path.basename(funsd_json_path).replace('.json', '.png')
    
    output_dict = {
        "image_path": image_filename,
        "words": all_words,
        "bboxes": all_bboxes,
        "entities": final_entities,
        "relations": final_relations
    }
    
    return output_dict

def main():
    for base_dir in [config.TRAIN_PATH, config.VAL_PATH]:
        funsd_dir = os.path.join(base_dir, 'annotations')
        output_file = os.path.join(base_dir, 'data.jsonl')
        if 'train' in base_dir:
            config.TRAIN_DATA_PATH = output_file
            print (f"Converting FUNSD training data from {funsd_dir} to {output_file}")
        else:
            config.VAL_DATA_PATH = output_file
            print (f"Converting FUNSD validation data from {funsd_dir} to {output_file}")


        if not os.path.isdir(funsd_dir):
            print(f"Error: Directory not found at {funsd_dir}")
            return

        json_files = [f for f in os.listdir(funsd_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {funsd_dir}")
            return

        print(f"Found {len(json_files)} JSON files. Starting conversion...")

        successful_conversions = 0
        total_entities = 0
        total_relations = 0

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in tqdm(json_files, desc="Converting files"):
                json_path = os.path.join(funsd_dir, filename)
                try:
                    converted_data = convert_funsd_to_jsonl(json_path)
                    outfile.write(json.dumps(converted_data) + '\n')
                    successful_conversions += 1
                    total_entities += len(converted_data.get('entities', []))
                    total_relations += len(converted_data.get('relations', []))
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        print(f"Conversion complete. Successfully converted {successful_conversions} out of {len(json_files)} files.")
        print(f"Total entities: {total_entities}, total relations: {total_relations}")
        
        if successful_conversions > 0:
            validate_jsonl_file(output_file)
        print ('-'*50 + '\n')

if __name__ == '__main__':
    main()