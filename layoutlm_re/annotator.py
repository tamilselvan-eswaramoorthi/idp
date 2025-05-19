import fitz  # PyMuPDF
import json
import os
import uuid

def extract_words_from_pdf(pdf_path):
    """
    Extracts words and their bounding boxes from each page of a PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of pages. Each page is a list of dictionaries,
              where each dictionary represents a word with 'text' and 'bbox' keys.
              bbox is (x0, y0, x1, y1).
    """
    doc_data = []
    try:
        doc = fitz.open(pdf_path)
        print(f"Processing PDF: {pdf_path}, Pages: {len(doc)}")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Using get_text("words") gives [(x0, y0, x1, y1, "word", block_no, line_no, word_no), ...]
            # We scale coordinates to a common basis like 1000x1000 if needed,
            # but LayoutLM often handles original coordinates well.
            # Let's keep original coordinates for now.
            words_on_page = page.get_text("words", sort=True) # Sort by y then x coordinate
            page_words = []
            for i, w in enumerate(words_on_page):
                x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
                # Basic cleaning - skip empty strings if any (unlikely with 'words')
                if text.strip():
                    page_words.append({
                        "id": i, # Simple index within the page
                        "text": text,
                        "bbox": [round(x0), round(y0), round(x1), round(y1)] # Use integers for simplicity
                    })
            doc_data.append(page_words)
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return doc_data

def display_words_for_annotation(page_words):
    """Displays words with their indices for easier selection."""
    print("\n--- Words on Page ---")
    for i, word_info in enumerate(page_words):
        print(f"[{i:03d}] {word_info['text']} (bbox: {word_info['bbox']})")
    print("---------------------\n")

def annotate_page_interactively(page_words, page_num):
    """
    Runs an interactive command-line loop to annotate entities and relations
    for a single page.

    Args:
        page_words (list): List of word dictionaries for the page.
        page_num (int): The current page number (0-based).

    Returns:
        tuple: A tuple containing two lists: (entities, relations) for the page.
               Returns ([], []) if no annotations are made or an error occurs.
    """
    entities = []
    relations = []
    entity_id_counter = 0

    if not page_words:
        print(f"Page {page_num + 1}: No words extracted, skipping annotation.")
        return [], []

    print(f"\n--- Annotating Page {page_num + 1} ---")
    display_words_for_annotation(page_words)

    # --- Entity Annotation Loop ---
    while True:
        action = input("Annotate Entity? (y/n): ").lower()
        if action != 'y':
            break

        while True:
            try:
                indices_str = input("Enter start and end word indices (e.g., '5 7' for words 5, 6, 7): ")
                start_idx, end_idx = map(int, indices_str.split())
                if 0 <= start_idx <= end_idx < len(page_words):
                    break
                else:
                    print(f"Invalid indices. Must be between 0 and {len(page_words) - 1}.")
            except ValueError:
                print("Invalid input. Please enter two numbers separated by space.")
            except Exception as e:
                 print(f"An error occurred: {e}")

        entity_label = input("Enter entity label (e.g., QUESTION, ANSWER, HEADER): ").upper()
        if not entity_label:
            print("Entity label cannot be empty. Skipping this entity.")
            continue

        # Aggregate text and calculate bounding box for the entity span
        entity_words = page_words[start_idx : end_idx + 1]
        entity_text = " ".join([w['text'] for w in entity_words])
        
        # Calculate bounding box covering all words in the span
        min_x0 = min(w['bbox'][0] for w in entity_words)
        min_y0 = min(w['bbox'][1] for w in entity_words)
        max_x1 = max(w['bbox'][2] for w in entity_words)
        max_y1 = max(w['bbox'][3] for w in entity_words)
        entity_bbox = [min_x0, min_y0, max_x1, max_y1]
        
        # Word indices included in this entity
        word_indices = list(range(start_idx, end_idx + 1))

        entity = {
            "id": entity_id_counter,
            "label": entity_label,
            "text": entity_text,
            "box": entity_bbox, # Overall box for the entity
            "word_indices": word_indices # Indices of words making up the entity
        }
        entities.append(entity)
        print(f"Added Entity: ID={entity['id']}, Label={entity['label']}, Text='{entity['text']}'")
        entity_id_counter += 1

    if not entities:
        print("No entities defined for this page.")
        return [], []

    # --- Relation Annotation Loop ---
    print("\n--- Current Entities ---")
    for entity in entities:
        print(f"ID={entity['id']}, Label={entity['label']}, Text='{entity['text']}'")
    print("------------------------\n")

    while True:
        action = input("Annotate Relation? (y/n): ").lower()
        if action != 'y':
            break

        while True:
            try:
                ids_str = input("Enter Head entity ID and Tail entity ID (e.g., '0 1'): ")
                head_id, tail_id = map(int, ids_str.split())
                # Check if these IDs exist
                if any(e['id'] == head_id for e in entities) and \
                   any(e['id'] == tail_id for e in entities) and \
                   head_id != tail_id:
                    break
                else:
                    print("Invalid entity IDs. Please choose from the list above and ensure head != tail.")
            except ValueError:
                print("Invalid input. Please enter two numbers separated by space.")
            except Exception as e:
                 print(f"An error occurred: {e}")


        relation_label = input("Enter relation label (e.g., CONTAINS, ASKS, ANSWERS): ").upper()
        if not relation_label:
            print("Relation label cannot be empty. Skipping this relation.")
            continue

        relation = {
            "head": head_id,
            "tail": tail_id,
            "label": relation_label
        }
        relations.append(relation)
        print(f"Added Relation: Head={relation['head']} --[{relation['label']}]--> Tail={relation['tail']}")

    return entities, relations

def save_annotations(doc_annotations, output_path):
    """
    Saves the annotations for the entire document in JSONL format.
    Each line is a JSON object representing one page's data and annotations.

    Args:
        doc_annotations (list): A list where each element corresponds to a page
                                and contains {'page': int, 'words': list,
                                'entities': list, 'relations': list}.
        output_path (str): Path to save the JSONL file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for page_data in doc_annotations:
                # Add a unique ID for the page/document slice
                page_data['id'] = str(uuid.uuid4()) 
                f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
        print(f"Annotations successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving annotations: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    pdf_file_path = 'data/5015-r-fill-24e.pdf'
    output_file_path = 'annotations.jsonl'

    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found at {pdf_file_path}")
    else:
        # 1. Extract words and boxes from PDF
        document_words = extract_words_from_pdf(pdf_file_path)
        
        all_page_annotations = []

        # 2. Annotate each page interactively
        for page_num, page_words in enumerate(document_words):
            entities, relations = annotate_page_interactively(page_words, page_num)
            
            # Prepare data structure for this page
            page_annotation_data = {
                "page": page_num + 1, # 1-based page number
                "words": page_words, # List of {'id': int, 'text': str, 'bbox': list}
                "entities": entities, # List of {'id': int, 'label': str, 'text': str, 'box': list, 'word_indices': list}
                "relations": relations # List of {'head': int, 'tail': int, 'label': str}
            }
            all_page_annotations.append(page_annotation_data)

        # 3. Save all annotations to a file
        if all_page_annotations:
             save_annotations(all_page_annotations, output_file_path)
        else:
             print("No annotations were generated.")