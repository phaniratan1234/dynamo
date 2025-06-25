# Fixed Loss Functions for DYNAMO

import torch
import torch.nn as nn


class QASpanLoss(nn.Module):
    """Simplified QA loss for start/end position prediction."""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # predictions: [batch_size, seq_len, 2] - start/end logits
        # targets: [batch_size, 2] - start/end positions
        
        start_logits = predictions[:, :, 0]  # [batch_size, seq_len]
        end_logits = predictions[:, :, 1]    # [batch_size, seq_len]
        
        start_targets = targets[:, 0]  # [batch_size]
        end_targets = targets[:, 1]    # [batch_size]
        
        start_loss = self.loss_fn(start_logits, start_targets)
        end_loss = self.loss_fn(end_logits, end_targets)
        
        return (start_loss + end_loss) / 2



class SummarizationSimplifiedLoss(nn.Module):
    """Simplified summarization loss using representation similarity."""
    
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # predictions: [batch_size, hidden_size] - summary representation
        # targets: [batch_size, hidden_size] - target summary representation
        
        # Use cosine similarity loss (1 - cosine_similarity)
        similarity = self.cos_sim(predictions, targets)
        loss = 1 - similarity.mean()
        
        return loss
