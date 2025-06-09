# model.py
import torch
import torch.nn as nn
from transformers import LayoutLMv3Model
import config
from utils import get_entity_embeddings, compute_variance_loss

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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                bbox=None,
                pixel_values=None,
                entity_token_indices=None, # Indices of first token for each entity
                relation_matrix=None # For RE loss calculation
                ):

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
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


        # --- Loss Calculation ---
        loss = self.calculate_loss(re_logits, relation_matrix, num_entities)
        
        # Check for NaN in loss and replace with zero if needed
        if loss is not None and torch.isnan(loss):
            print("Warning: NaN loss detected, setting to zero")
            loss = torch.tensor(0.0, device=re_logits.device, requires_grad=True)
            
        return loss, re_logits

    def calculate_loss(self, re_logits, relation_matrix, num_entities):
        loss = None

        # RE Loss (Binary Cross Entropy on the matrix)
        if relation_matrix is not None:
            loss_fct_re = nn.BCEWithLogitsLoss(reduction='mean')
            # Only compute loss if num_entities > 0
            if num_entities > 0:
                # Add a small epsilon to prevent extreme values
                re_logits = torch.clamp(re_logits, min=-100, max=100)
                loss = loss_fct_re(re_logits, relation_matrix)
            else:
                loss = torch.tensor(0.0, device=re_logits.device, requires_grad=True)

        # Variance loss (RSF - Relation-Specific Features)
        if num_entities > 1:  # Need at least 2 entities for meaningful variance
            try:
                re_variance_loss = compute_variance_loss(re_logits, tau=config.RSF_TAU)
                # Check for NaN in variance loss
                if torch.isnan(re_variance_loss):
                    print("Warning: NaN in variance loss")
                    re_variance_loss = torch.tensor(0.0, device=re_logits.device, requires_grad=True)
                elif re_variance_loss > 100:  # Clip extremely large variance loss
                    re_variance_loss = torch.tensor(100.0, device=re_logits.device, requires_grad=True)
                    
                if loss is not None:
                    loss = loss + config.RSF_LOSS_WEIGHT * re_variance_loss
            except Exception as e:
                print(f"Error in variance loss calculation: {e}")
                # Continue training even if variance loss fails

        return loss