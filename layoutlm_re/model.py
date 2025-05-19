# model.py
import torch
import torch.nn as nn
from transformers import LayoutLMv3Model
import config
from utils import get_entity_embeddings

class LayoutLMv3ForRelationExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layoutlmv3 = LayoutLMv3Model.from_pretrained(
            config.MODEL_NAME,
            cache_dir=config.CACHE_DIR,
        )
        self.dropout = nn.Dropout(self.layoutlmv3.config.hidden_dropout_prob)

        hidden_size = self.layoutlmv3.config.hidden_size
        re_head_input_size = hidden_size

        # Relation Extraction Head (Bilinear)
        # Takes two entity embeddings and predicts relation score
        self.bilinear_re_head = nn.Bilinear(
            re_head_input_size, re_head_input_size, 1 # Output 1 logit per pair
        )

        # Entity Extraction Head (Linear Classifier for IOB tags)
        if config.USE_EEF:
            self.linear_ee_head = nn.Linear(hidden_size, len(config.ENTITY_LABELS))

    def forward(self,
                input_ids=None,
                attention_mask=None,
                bbox=None,
                pixel_values=None,
                entity_token_indices=None, # Indices of first token for each entity
                ee_labels=None, # For EEF loss calculation
                relation_matrix=None # For RE loss calculation
                ):

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values if config.USE_IMAGE else None,
            output_hidden_states=False,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # --- Relation Extraction ---
        # 1. Get embeddings for each entity
        entity_embeds = get_entity_embeddings(sequence_output, entity_token_indices, pooling=config.ENTITY_POOLING)
        # entity_embeds shape: (batch_size, num_entities, hidden_size)

        batch_size, num_entities, hidden_size = entity_embeds.shape
        re_logits = torch.zeros((batch_size, num_entities, num_entities), device=entity_embeds.device)

        if num_entities > 0:
             # 2. Use Bilinear layer for relation prediction
             # Expand dims to compute scores for all pairs (i, j)
             # Head entities: (batch_size, num_entities, 1, hidden_size_re) -> Expand tail dim
             # Tail entities: (batch_size, 1, num_entities, hidden_size_re) -> Expand head dim
             head_embeds = entity_embeds.unsqueeze(2).expand(-1, -1, num_entities, -1)
             tail_embeds = entity_embeds.unsqueeze(1).expand(-1, num_entities, -1, -1)

             # Bilinear expects (N, *, H1) and (N, *, H2) -> Outputs (N, *, O)
             # Reshape to (batch_size * num_entities * num_entities, hidden_size_re)
             re_head_input_size = entity_embeds.shape[-1]
             head_embeds_flat = head_embeds.reshape(-1, re_head_input_size)
             tail_embeds_flat = tail_embeds.reshape(-1, re_head_input_size)

             # Compute logits
             re_logits_flat = self.bilinear_re_head(head_embeds_flat, tail_embeds_flat).squeeze(-1)
             re_logits = re_logits_flat.reshape(batch_size, num_entities, num_entities)

        # --- Entity Extraction (if enabled) ---
        ee_logits = None
        if config.USE_EEF:
            ee_logits = self.linear_ee_head(sequence_output) # (batch_size, seq_len, num_ee_labels)

        # --- Loss Calculation ---
        output = self.calculate_loss(re_logits, ee_logits, relation_matrix, ee_labels, num_entities)
        return output

    def calculate_loss(self, re_logits, ee_logits, relation_matrix, ee_labels, num_entities):
        loss = None
        re_loss = None
        ee_loss = None

        # RE Loss (Binary Cross Entropy on the matrix)
        if relation_matrix is not None:
            loss_fct_re = nn.BCEWithLogitsLoss()
            # Only compute loss if num_entities > 0
            if num_entities > 0:
                re_loss = loss_fct_re(re_logits, relation_matrix)
            else:
                re_loss = torch.tensor(0.0, device=re_logits.device) # Or handle appropriately
            loss = re_loss

        # EE Loss (Cross Entropy on token labels)
        if config.USE_EEF and ee_labels is not None:
            loss_fct_ee = nn.CrossEntropyLoss() # Ignores -100 labels
            # Flatten logits and labels for CE loss: (batch_size * seq_len, num_labels), (batch_size * seq_len,)
            ee_loss = loss_fct_ee(ee_logits.view(-1, len(config.ENTITY_LABELS)), ee_labels.view(-1))
            if loss is not None:
                loss += config.EE_LOSS_WEIGHT * ee_loss
            else:
                loss = config.EE_LOSS_WEIGHT * ee_loss

        # TODO: Add variance loss if config.USE_RSF_LOSS is True

        output = {"loss": loss, "re_loss": re_loss, "ee_loss": ee_loss, "re_logits": re_logits, "ee_logits": ee_logits}
        return output