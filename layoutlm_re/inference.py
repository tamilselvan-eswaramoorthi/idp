import random
from PIL import Image, ImageDraw

import torch
import pymupdf  # Fitz
import numpy as np
from transformers import LayoutLMv3TokenizerFast, AutoImageProcessor

import config
from model import LayoutLMv3ForRelationExtraction
from utils import normalize_bbox, get_entity_token_indices, apply_rsf

# Ensure all required source files are in the same directory or accessible
# (config.py, model.py, utils.py)

def find_entities_and_words_from_pdf(page):
    """
    Extracts words and their bounding boxes from a PDF page and
    uses a heuristic (grouping by text blocks) to identify entities.

    Args:
        page (pymupdf.Page): The PDF page object.

    Returns:
        tuple: A tuple containing:
            - list: A list of all word strings.
            - list: A list of corresponding unnormalized bounding boxes.
            - list: A list of detected entity dictionaries.
    """
    # Extract words with block information
    # Format: [x0, y0, x1, y1, "word", block_no, line_no, word_no]
    word_list = page.get_text("words")
    if not word_list:
        return [], [], []

    words = [w[4] for w in word_list]
    unnormalized_bboxes = [list(w[:4]) for w in word_list]

    # Group words by their block number to form entities
    blocks = {}
    for i, word_info in enumerate(word_list):
        block_num = word_info[5]
        if block_num not in blocks:
            blocks[block_num] = []
        blocks[block_num].append(i)

    # Create entity list from the detected blocks
    entities = []
    entity_id_counter = 0
    for block_num, word_indices in sorted(blocks.items()):
        if not word_indices:
            continue

        start_word_idx = min(word_indices)
        end_word_idx = max(word_indices)
        
        # Merge bboxes of all words in the entity to get a single entity bbox
        entity_bboxes = [unnormalized_bboxes[i] for i in word_indices]
        min_x = min(b[0] for b in entity_bboxes)
        min_y = min(b[1] for b in entity_bboxes)
        max_x = max(b[2] for b in entity_bboxes)
        max_y = max(b[3] for b in entity_bboxes)
        merged_bbox = [min_x, min_y, max_x, max_y]
        
        # Get the full text of the entity
        entity_text = " ".join(words[idx] for idx in sorted(word_indices))

        entities.append({
            'id': entity_id_counter,
            'label': 'ENTITY',  # Use a generic label as we don't know the type
            'span': [start_word_idx, end_word_idx],
            'text': entity_text,
            'bbox': merged_bbox  # Store for visualization
        })
        entity_id_counter += 1

    return words, unnormalized_bboxes, entities


def prepare_inference_data(tokenizer, image_processor, image, words, unnormalized_boxes, entities):
    """
    Prepares the extracted PDF data into the format required by the model,
    replicating the logic from VRDDataset.
    """
    width, height = image.size

    # --- Bounding Box Optimization (BBO) ---
    # Merge boxes for each entity and assign the merged box to all its words
    merged_boxes = unnormalized_boxes.copy()
    for entity in entities:
        start, end = entity['span']
        entity_boxes = unnormalized_boxes[start:end+1]
        if not entity_boxes: continue
        min_x = min(box[0] for box in entity_boxes)
        min_y = min(box[1] for box in entity_boxes)
        max_x = max(box[2] for box in entity_boxes)
        max_y = max(box[3] for box in entity_boxes)
        merged_box = [min_x, min_y, max_x, max_y]
        for i in range(start, end + 1):
            merged_boxes[i] = merged_box
    
    unnormalized_boxes = merged_boxes
    normalized_boxes = [normalize_bbox(box, width, height) for box in unnormalized_boxes]

    # --- Entity Marking (EM) ---
    # Insert a generic entity marker before each entity's text
    entity_starts = {e['span'][0]: e['label'] for e in entities}
    new_words, new_boxes = [], []
    word_idx_map = {}
    current_new_idx = 0
    for i in range(len(words)):
        marker_insertion_point = current_new_idx
        if i in entity_starts:
            new_words.append("entity") # Use a generic marker
            new_boxes.append(normalized_boxes[i])
            current_new_idx += 1
        
        word_insertion_point = current_new_idx
        new_words.append(words[i])
        new_boxes.append(normalized_boxes[i])
        current_new_idx += 1
        word_idx_map[i] = word_insertion_point

    words, normalized_boxes = new_words, new_boxes
    
    # Update entity spans to reflect the new word indices after EM
    updated_entity_word_indices = []
    for entity in entities:
        original_start, original_end = entity['span']
        if original_start in word_idx_map and original_end in word_idx_map:
            new_start = word_idx_map[original_start]
            new_end = word_idx_map[original_end]
            updated_entity_word_indices.append([new_start, new_end])

    # --- Tokenization & Feature Creation ---
    encoding = tokenizer(
        words,
        boxes=normalized_boxes,
        max_length=config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    word_ids = encoding.word_ids(batch_index=0)
    image_features = image_processor(image, return_tensors="pt")
    encoding['pixel_values'] = image_features['pixel_values']

    # --- Entity Token Indices ---
    actual_entity_first_token_indices = get_entity_token_indices(word_ids, updated_entity_word_indices)
    padded_entity_token_indices = torch.full(
        (config.MAX_ENTITIES,), config.ENTITY_INDEX_PAD_VALUE, dtype=torch.long
    )
    
    num_found_indices = len(actual_entity_first_token_indices)
    if num_found_indices > 0:
        copy_len = min(config.MAX_ENTITIES, num_found_indices)
        padded_entity_token_indices[:copy_len] = torch.tensor(actual_entity_first_token_indices[:copy_len], dtype=torch.long)
    
    encoding['entity_token_indices'] = padded_entity_token_indices.unsqueeze(0)
    
    # Unsqueeze all tensors to create a batch of 1
    final_batch = {k: v.to(config.DEVICE) for k, v in encoding.items()}
    return final_batch


def draw_relations_on_image(image, entities, relations_matrix):
    """Draws arrows on the image to visualize predicted relations."""
    draw = ImageDraw.Draw(image, "RGBA")
    
    # Define colors for arrows
    colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#A133FF", "#33FFA1"]
    
    for i in range(relations_matrix.shape[0]):
        for j in range(relations_matrix.shape[1]):
            if relations_matrix[i, j] == 1:
                # This is a predicted relation from entity i to entity j
                head_entity = entities[i]
                tail_entity = entities[j]

                head_bbox = head_entity['bbox']
                tail_bbox = tail_entity['bbox']

                # Calculate center points of the bounding boxes
                head_center = ((head_bbox[0] + head_bbox[2]) / 2, (head_bbox[1] + head_bbox[3]) / 2)
                tail_center = ((tail_bbox[0] + tail_bbox[2]) / 2, (tail_bbox[1] + tail_bbox[3]) / 2)
                
                color = random.choice(colors)
                
                # Draw a thicker line for the arrow
                draw.line([head_center, tail_center], fill=color, width=4)

                # Draw a circle at the head and a smaller one at the tail to indicate direction
                r_head = 8
                draw.ellipse((head_center[0]-r_head, head_center[1]-r_head, head_center[0]+r_head, head_center[1]+r_head), fill=color)
                r_tail = 4
                draw.ellipse((tail_center[0]-r_tail, tail_center[1]-r_tail, tail_center[0]+r_tail, tail_center[1]+r_tail), fill=color, outline="black")

    return image


def main(model_path, pdf_path, output_path):
    # --- 1. Load Model, Tokenizer, and Image Processor ---
    print("Loading model and tokenizers...")
    device = config.DEVICE
    model = LayoutLMv3ForRelationExtraction().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(config.MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(config.IMAGE_TOKENIZER_NAME)

    # --- 2. Process PDF ---
    print(f"Processing PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)  # Process the first page

    # Render page to an image
    pix = page.get_pixmap(dpi=200)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGB")

    # Extract words and heuristic entities
    words, unnormalized_boxes, entities = find_entities_and_words_from_pdf(page)
    
    if not entities:
        print("No text blocks (entities) found on the first page. Exiting.")
        return
        
    num_entities = len(entities)
    print(f"Found {num_entities} potential entities (text blocks).")

    if num_entities > config.MAX_ENTITIES:
        print(f"Warning: Found {num_entities} entities, but model supports max {config.MAX_ENTITIES}. Truncating.")
        entities = entities[:config.MAX_ENTITIES]
        num_entities = len(entities)

    # --- 3. Prepare Data for Inference ---
    batch = prepare_inference_data(tokenizer, image_processor, image, words, unnormalized_boxes, entities)

    # --- 4. Run Inference ---
    print("Running inference...")
    with torch.no_grad():
        _, re_logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            bbox=batch['bbox'],
            pixel_values=batch['pixel_values'],
            entity_token_indices=batch['entity_token_indices'],
            relation_matrix=None  # No labels during inference
        )

    # --- 5. Post-process Predictions ---
    # Squeeze batch dimension and slice to the actual number of entities
    pred_probs = torch.sigmoid(re_logits.squeeze(0)[:num_entities, :num_entities])
    
    # Apply Relation-Specific Features (RSF) filtering
    preds_after_rsf = apply_rsf(pred_probs, tau=config.RSF_TAU)
    print (preds_after_rsf)
    
    # Apply threshold to get final binary predictions
    final_preds = (preds_after_rsf > 0.1).int().cpu().numpy()
    
    # Disallow self-relations
    np.fill_diagonal(final_preds, 0)
    
    # --- 6. Print and Visualize Results ---
    print("\n--- Predicted Relations ---")
    relation_found = False
    for i in range(num_entities):
        for j in range(num_entities):
            if final_preds[i, j] == 1:
                head_text = entities[i]['text'].replace('\n', ' ').strip()
                tail_text = entities[j]['text'].replace('\n', ' ').strip()
                print(f"  FROM: \"{head_text}\" \n  --> TO: \"{tail_text}\"\n")
                relation_found = True
    
    if not relation_found:
        print("No relations found meeting the threshold.")

    # Draw relations on the image
    output_image = draw_relations_on_image(image.copy(), entities, final_preds)
    
    # Save the output image
    output_image.save(output_path)
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    pdf_path = 'f1040.pdf'
    output_path = 'output/inference_result.png'
    model_path = 'output/best_model.pt'
    main(model_path, pdf_path, output_path)