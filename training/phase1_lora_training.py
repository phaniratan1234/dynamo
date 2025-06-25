"""
Phase 1 Training: Individual LoRA Adapter Training
Trains each LoRA adapter independently on task-specific data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any
import os
from tqdm import tqdm
import wandb

from model import DynamoModel
from data import DatasetLoader
from training.losses import TaskSpecificLoss
from utils.config import Config
from utils.logger import get_logger, TrainingLogger
from utils.helpers import set_seed, count_parameters, move_to_device, AverageMeter, EarlyStopping

logger = get_logger(__name__)


class Phase1Trainer:
    """
    Trainer for Phase 1: Individual LoRA adapter training.
    """
    
    def __init__(self, config: Config, model: DynamoModel):
        """
        Initialize Phase 1 trainer.
        
        Args:
            config: Training configuration
            model: DYNAMO model
        """
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        
        # Set model to Phase 1 mode
        self.model.set_training_phase("phase1")
        self.model.to(self.device)
        
        # Initialize data loader
        self.data_loader = DatasetLoader(config.__dict__)
        
        # Training state
        self.current_task = None
        self.current_epoch = 0
        self.global_step = 0
        
        # Logging
        self.training_logger = TrainingLogger(config.log_dir)
        
        logger.info("Phase 1 trainer initialized")
    
    def train_single_adapter(
        self,
        task_name: str,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = None,
        learning_rate: float = None
    ) -> Dict[str, float]:
        """
        Train a single LoRA adapter for a specific task.
        
        Args:
            task_name: Name of the task
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Training metrics
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        if learning_rate is None:
            learning_rate = self.config.training.lora_lr
        
        self.current_task = task_name
        logger.info(f"Training {task_name} adapter for {num_epochs} epochs")
        
        # Get the specific adapter
        adapter = self.model.adapters.get_adapter(task_name)
        
        # Freeze all other adapters
        self.model.adapters.freeze_adapters()
        self.model.adapters.unfreeze_adapters([task_name])
        
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer(adapter, learning_rate)
        scheduler = self._setup_scheduler(optimizer, len(train_dataloader) * num_epochs)
        
        # Setup loss function
        loss_fn = TaskSpecificLoss(task_name)
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=getattr(self.config.training, 'patience', 3),
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Training metrics
        best_val_loss = float('inf')
        training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf')
        }
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.training_logger.log_epoch_start(epoch)
            
            # Training phase
            train_loss = self._train_epoch(
                train_dataloader, optimizer, scheduler, loss_fn, task_name
            )
            training_metrics['train_loss'].append(train_loss)
            
            # Validation phase
            if val_dataloader is not None:
                val_loss = self._validate_epoch(val_dataloader, loss_fn, task_name)
                training_metrics['val_loss'].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    training_metrics['best_val_loss'] = best_val_loss
                    self._save_adapter_checkpoint(task_name, epoch, val_loss)
                
                # Early stopping check
                if early_stopping(val_loss, adapter):
                    logger.info(f"Early stopping triggered for {task_name} at epoch {epoch}")
                    break
                
                self.training_logger.log_epoch_end(epoch, train_loss, val_loss=val_loss)
            else:
                self.training_logger.log_epoch_end(epoch, train_loss)
        
        logger.info(f"Completed training {task_name} adapter. Best val loss: {best_val_loss:.4f}")
        return training_metrics
    
    def train_all_adapters(self) -> Dict[str, Dict[str, float]]:
        """
        Train all LoRA adapters sequentially.
        
        Returns:
            Training metrics for all adapters
        """
        logger.info("Starting Phase 1: Training all LoRA adapters")
        
        # Load datasets
        train_datasets = self.data_loader.create_datasets('train')
        val_datasets = self.data_loader.create_datasets('validation')
        
        # Create data loaders
        train_dataloaders = self.data_loader.create_dataloaders(
            train_datasets,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        val_dataloaders = self.data_loader.create_dataloaders(
            val_datasets,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False
        )
        
        all_metrics = {}
        
        # Check for existing checkpoints and skip completed tasks
        completed_tasks = self._check_existing_checkpoints()
        if completed_tasks:
            logger.info(f"Found existing checkpoints for tasks: {completed_tasks}")
            for task in completed_tasks:
                logger.info(f"✅ Skipping {task} - checkpoint already exists")
        
        # Train each adapter
        for task_name in self.model.task_names:
            # Skip if checkpoint already exists
            if task_name in completed_tasks:
                logger.info(f"⏭️  Skipping {task_name} adapter - already trained")
                continue
                
            logger.info(f"{'='*50}")
            logger.info(f"Training {task_name} adapter")
            logger.info(f"{'='*50}")
            
            if task_name in train_dataloaders:
                train_dl = train_dataloaders[task_name]
                val_dl = val_dataloaders.get(task_name, None)
                
                metrics = self.train_single_adapter(
                    task_name, train_dl, val_dl
                )
                all_metrics[task_name] = metrics
                
                # Log to wandb if enabled
                if self.config.use_wandb:
                    self._log_wandb_metrics(task_name, metrics)
            else:
                logger.warning(f"No training data found for task: {task_name}")
        
        # Save final model
        self._save_phase1_checkpoint(all_metrics)
        
        logger.info("Phase 1 training completed")
        return all_metrics
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
        task_name: str
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        
        progress_bar = tqdm(dataloader, desc=f"Training {task_name}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = move_to_device(batch, self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task_labels=torch.full((batch['input_ids'].size(0),), 
                                     self.model.task_to_idx[task_name], 
                                     device=self.device)
            )
            
            # Get task-specific output
            if task_name in outputs['task_outputs']:
                predictions = outputs['task_outputs'][task_name]
                targets = batch['target']
                
                # Compute loss
                loss = loss_fn(predictions, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                loss_meter.update(loss.item(), batch['input_ids'].size(0))
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.evaluation.logging_steps == 0:
                    self.training_logger.log_step(
                        loss.item(),
                        scheduler.get_last_lr()[0]
                    )
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        return loss_meter.avg
    
    def _validate_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        task_name: str
    ) -> float:
        """Validate for one epoch."""
        self.model.eval()
        loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validating {task_name}"):
                batch = move_to_device(batch, self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_labels=torch.full((batch['input_ids'].size(0),), 
                                         self.model.task_to_idx[task_name], 
                                         device=self.device)
                )
                
                # Get task-specific output
                if task_name in outputs['task_outputs']:
                    predictions = outputs['task_outputs'][task_name]
                    targets = batch['target']
                    
                    # Compute loss
                    loss = loss_fn(predictions, targets)
                    loss_meter.update(loss.item(), batch['input_ids'].size(0))
        
        return loss_meter.avg
    
    def _setup_optimizer(self, adapter: nn.Module, learning_rate: float) -> optim.Optimizer:
        """Setup optimizer for adapter training."""
        # Only optimize the current adapter parameters
        optimizer = optim.AdamW(
            adapter.parameters(),
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        return optimizer
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, total_steps: int) -> Any:
        """Setup learning rate scheduler."""
        warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler
    
    def _save_adapter_checkpoint(self, task_name: str, epoch: int, val_loss: float):
        """Save adapter checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase1", task_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'adapter_state_dict': self.model.adapters.get_adapter(task_name).state_dict()
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"best_adapter.pt")
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved {task_name} adapter checkpoint to {checkpoint_path}")
    
    def _save_phase1_checkpoint(self, all_metrics: Dict[str, Dict[str, float]]):
        """Save complete Phase 1 checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase1")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save all adapters
        self.model.adapters.save_adapters(checkpoint_dir)
        
        # Save training metrics
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.pt")
        torch.save(all_metrics, metrics_path)
        
        logger.info(f"Saved Phase 1 checkpoint to {checkpoint_dir}")
    
    def _check_existing_checkpoints(self) -> List[str]:
        """Check for existing adapter checkpoints and return list of completed tasks."""
        completed_tasks = []
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase1")
        
        for task_name in self.model.task_names:
            task_checkpoint_path = os.path.join(checkpoint_dir, task_name, "best_adapter.pt")
            if os.path.exists(task_checkpoint_path):
                completed_tasks.append(task_name)
                # Load the checkpoint to restore the adapter weights
                try:
                    checkpoint = torch.load(task_checkpoint_path, map_location=self.device, weights_only=False)
                    adapter = self.model.adapters.get_adapter(task_name)
                    adapter.load_state_dict(checkpoint['adapter_state_dict'])
                    logger.info(f"Loaded {task_name} adapter from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load {task_name} checkpoint: {e}")
        
        return completed_tasks
    
    def _log_wandb_metrics(self, task_name: str, metrics: Dict[str, float]):
        """Log metrics to Weights & Biases."""
        if not self.config.use_wandb:
            return
        
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                wandb_metrics[f"phase1/{task_name}/{key}"] = value[-1]
            elif isinstance(value, (int, float)):
                wandb_metrics[f"phase1/{task_name}/{key}"] = value
        
        wandb.log(wandb_metrics, step=self.global_step)


def run_phase1_training(config: Config, model: DynamoModel) -> Dict[str, Dict[str, float]]:
    """
    Run Phase 1 training.
    
    Args:
        config: Training configuration
        model: DYNAMO model
    
    Returns:
        Training metrics for all adapters
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize trainer
    trainer = Phase1Trainer(config, model)
    
    # Run training
    metrics = trainer.train_all_adapters()
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from utils.config import get_config
    from model import DynamoModel
    
    # Load configuration
    config = get_config()
    
    # Initialize model
    model = DynamoModel(config.__dict__)
    
    # Run Phase 1 training
    metrics = run_phase1_training(config, model)
    
    print("Phase 1 training completed!")
    for task_name, task_metrics in metrics.items():
        print(f"{task_name}: Best val loss = {task_metrics['best_val_loss']:.4f}")

