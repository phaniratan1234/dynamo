# Simplified Task Heads for DYNAMO


def _create_task_head(self) -> nn.Module:
    """Create task-specific prediction head - SIMPLIFIED VERSION."""
    
    if self.task_name == "sentiment":
        # Binary sentiment classification
        return nn.Sequential(
            nn.Linear(self.hidden_size, 2)  # Positive/Negative
        )
    
    elif self.task_name == "qa":
        # Question answering - span prediction only
        return nn.Linear(self.hidden_size, 2)  # Start and end logits
    
    elif self.task_name in ["summarization", "code_generation", "translation"]:
        # Simplified: Use representation learning instead of generation
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)  # Same size representation
        )
    
    else:
        # Generic classification
        return nn.Linear(self.hidden_size, 1)



def forward(self, hidden_states: torch.Tensor, task_type: str = None) -> torch.Tensor:
    """Forward pass through task head - SIMPLIFIED VERSION."""
    
    if self.task_name == "sentiment":
        # Use CLS token for classification
        cls_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        return self.task_head(cls_hidden)
    
    elif self.task_name == "qa":
        # Use all tokens for span prediction
        return self.task_head(hidden_states)  # [batch_size, seq_len, 2]
    
    else:
        # For generation tasks, use CLS token representation
        cls_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        return self.task_head(cls_hidden)  # [batch_size, hidden_size]
