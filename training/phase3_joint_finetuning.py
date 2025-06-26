"""
Phase 3 Training: Joint Fine-tuning
Joint optimization of router and adapters with curriculum learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any, Tuple
import os
from tqdm import tqdm
import wandb
import numpy as np
import random

from model import DynamoModel
from data import DatasetLoader, create_mixed_task_dataset, create_mixed_task_dataloader
from training.losses import DynamoLoss, create_loss_function, CurriculumLoss
from utils.config import Config
from utils.logger import get_logger, TrainingLogger
from utils.helpers import (
    set_seed, count_parameters, move_to_device, AverageMeter, 
    EarlyStopping, interpolate_configs
)

logger = get_logger(__name__)


class Phase3Trainer:
    """
    Trainer for Phase 3: Joint fine-tuning of router and adapters.
    """
    
    def __init__(self, config: Config, model: DynamoModel):
        """
        Initialize Phase 3 trainer.
        
        Args:
            config: Training configuration
            model: DYNAMO model
        """
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        
        # Set model to Phase 3 mode
        self.model.set_training_phase("phase3")
        self.model.to(self.device)
        
        # Initialize data loader
        self.data_loader = DatasetLoader(config.__dict__)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Curriculum learning parameters
        self.curriculum_start_ratio = config.training.curriculum_start_ratio
        self.curriculum_end_ratio = config.training.curriculum_end_ratio
        self.current_curriculum_ratio = self.curriculum_start_ratio
        
        # Joint training specifics
        self.adapter_lr_scale = 0.1  # Scale down adapter learning rate
        self.router_lr_scale = 1.0   # Keep router learning rate
        
        # Logging
        self.training_logger = TrainingLogger(config.log_dir)
        
        logger.info("Phase 3 trainer initialized")
    
    def joint_train(
        self,
        mixed_dataloader: DataLoader,
        single_task_dataloaders: Dict[str, DataLoader],
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = None,
        learning_rate: float = None
    ) -> Dict[str, float]:
        """
        Joint training of router and adapters.
        
        Args:
            mixed_dataloader: Mixed task training data loader
            single_task_dataloaders: Single task data loaders
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Base learning rate
        
        Returns:
            Training metrics
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        if learning_rate is None:
            learning_rate = self.config.training.joint_lr
        
        logger.info(f"Joint training for {num_epochs} epochs")
        
        # Setup optimizer and scheduler
        optimizer = self._setup_joint_optimizer(learning_rate)
        
        # Calculate total steps for both mixed and single task training
        total_steps = (len(mixed_dataloader) + sum(len(dl) for dl in single_task_dataloaders.values())) * num_epochs
        scheduler = self._setup_scheduler(optimizer, total_steps)
        
        # Setup loss function with curriculum learning
        base_loss_fn = create_loss_function(
            self.config.training.__dict__,
            self.model.task_names,
            self.model.adapters.get_num_tasks()
        )
        
        curriculum_schedule = self._create_curriculum_schedule(total_steps)
        loss_fn = CurriculumLoss(base_loss_fn, curriculum_schedule)
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=getattr(self.config.training, 'patience', 7),
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Training metrics
        best_val_loss = float('inf')
        training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'routing_accuracy': [],
            'task_performance': {task: [] for task in self.model.task_names},
            'load_balance_loss': [],
            'efficiency_loss': [],
            'consistency_loss': [],
            'best_val_loss': float('inf')
        }
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.training_logger.log_epoch_start(epoch)
            
            # Update curriculum
            self._update_curriculum(epoch, num_epochs)
            
            # Training phase with curriculum learning
            epoch_metrics = self._train_epoch_curriculum(
                mixed_dataloader, single_task_dataloaders, 
                optimizer, scheduler, loss_fn
            )
            
            # Update training metrics
            for key, value in epoch_metrics.items():
                if key in training_metrics:
                    if isinstance(training_metrics[key], list):
                        training_metrics[key].append(value)
                    elif isinstance(training_metrics[key], dict):
                        for sub_key, sub_value in value.items():
                            if sub_key in training_metrics[key]:
                                training_metrics[key][sub_key].append(sub_value)
            
            # Validation phase
            if val_dataloader is not None:
                val_metrics = self._validate_epoch(val_dataloader, base_loss_fn)
                
                val_loss = val_metrics['total_loss']
                training_metrics['val_loss'].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    training_metrics['best_val_loss'] = best_val_loss
                    self._save_joint_checkpoint(epoch, val_loss, val_metrics)
                
                # Early stopping check
                if early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                self.training_logger.log_epoch_end(
                    epoch, epoch_metrics['train_loss'], 
                    val_loss=val_loss,
                    routing_accuracy=val_metrics.get('routing_accuracy', 0.0)
                )
            else:
                self.training_logger.log_epoch_end(epoch, epoch_metrics['train_loss'])
            
            # Update curriculum loss step
            loss_fn.step()
        
        logger.info(f"Completed joint training. Best val loss: {best_val_loss:.4f}")
        return training_metrics
    
    def train_with_curriculum(self) -> Dict[str, float]:
        """
        Train with curriculum learning strategy.
        
        Returns:
            Training metrics
        """
        logger.info("Starting Phase 3: Joint fine-tuning with curriculum learning")
        
        # Load datasets
        train_datasets = self.data_loader.create_datasets('train')
        val_datasets = self.data_loader.create_datasets('validation')
        
        # Create single task data loaders
        single_task_dataloaders = self.data_loader.create_dataloaders(
            train_datasets,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        
        # Create mixed task dataset with curriculum
        mixed_train_dataset = create_mixed_task_dataset(
            train_datasets,
            self.config.__dict__,
            num_examples=self.config.data.mixed_task_size
        )
        
        mixed_val_dataset = create_mixed_task_dataset(
            val_datasets,
            self.config.__dict__,
            num_examples=self.config.data.mixed_task_size // 5
        )
        
        # Create mixed task data loaders
        mixed_train_dataloader = create_mixed_task_dataloader(
            mixed_train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        
        val_dataloader = create_mixed_task_dataloader(
            mixed_val_dataset,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False
        )
        
        # Joint training
        metrics = self.joint_train(
            mixed_train_dataloader, 
            single_task_dataloaders,
            val_dataloader
        )
        
        # Save final checkpoint
        self._save_phase3_checkpoint(metrics)
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            self._log_wandb_metrics(metrics)
        
        logger.info("Phase 3 training completed")
        return metrics
    
    def _train_epoch_curriculum(
        self,
        mixed_dataloader: DataLoader,
        single_task_dataloaders: Dict[str, DataLoader],
        optimizer: optim.Optimizer,
        scheduler: Any,
        loss_fn: CurriculumLoss
    ) -> Dict[str, Any]:
        """Train for one epoch with curriculum learning."""
        self.model.train()
        
        # Metrics tracking
        loss_meters = {
            'train_loss': AverageMeter(),
            'load_balance_loss': AverageMeter(),
            'efficiency_loss': AverageMeter(),
            'consistency_loss': AverageMeter(),
            'routing_accuracy': AverageMeter()
        }
        
        task_performance = {task: AverageMeter() for task in self.model.task_names}
        
        # Curriculum-based training strategy
        # Start with more single-task examples, gradually increase mixed-task examples
        
        # Calculate number of batches for each type
        total_mixed_batches = len(mixed_dataloader)
        mixed_batches_to_use = int(total_mixed_batches * (1 - self.current_curriculum_ratio))
        
        # Create iterators
        mixed_iter = iter(mixed_dataloader)
        single_task_iters = {task: iter(dl) for task, dl in single_task_dataloaders.items()}
        
        # Training with curriculum
        progress_bar = tqdm(
            total=mixed_batches_to_use + sum(len(dl) for dl in single_task_dataloaders.values()),
            desc=f"Joint Training (curriculum: {self.current_curriculum_ratio:.2f})"
        )
        
        # Mixed task training
        for batch_idx in range(mixed_batches_to_use):
            try:
                batch = next(mixed_iter)
                metrics = self._train_mixed_batch(batch, optimizer, scheduler, loss_fn)
                self._update_meters(loss_meters, task_performance, metrics)
                progress_bar.update(1)
            except StopIteration:
                break
        
        # Single task training (with curriculum weighting)
        single_task_batches_per_task = max(1, int(
            len(single_task_dataloaders[self.model.task_names[0]]) * self.current_curriculum_ratio
        ))
        
        for task_name in self.model.task_names:
            if task_name in single_task_iters:
                for batch_idx in range(min(single_task_batches_per_task, len(single_task_dataloaders[task_name]))):
                    try:
                        batch = next(single_task_iters[task_name])
                        metrics = self._train_single_task_batch(
                            batch, task_name, optimizer, scheduler, loss_fn
                        )
                        self._update_meters(loss_meters, task_performance, metrics)
                        progress_bar.update(1)
                    except StopIteration:
                        break
        
        progress_bar.close()
        
        # Prepare epoch metrics
        epoch_metrics = {key: meter.avg for key, meter in loss_meters.items()}
        epoch_metrics['task_performance'] = {task: meter.avg for task, meter in task_performance.items()}
        
        return epoch_metrics
    
    def _train_mixed_batch(
        self,
        batch: Dict[str, Any],
        optimizer: optim.Optimizer,
        scheduler: Any,
        loss_fn: CurriculumLoss
    ) -> Dict[str, float]:
        """Train on a mixed task batch."""
        batch = move_to_device(batch, self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_routing_info=True
        )
        
        # Prepare targets and compute loss
        task_targets = self._prepare_mixed_task_targets(batch)
        
        # Get backbone embeddings for consistency loss
        backbone_outputs = self.model.backbone(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        cls_embeddings = backbone_outputs.last_hidden_state[:, 0, :]
        
        # Compute losses
        losses = loss_fn(
            task_outputs=outputs['task_outputs'],
            task_targets=task_targets,
            routing_probs=outputs['routing_probs'],
            input_embeddings=cls_embeddings,
            temperature=outputs['routing_info'].get('temperature'),
            training_phase="phase3"
        )
        
        # Backward pass
        total_loss = losses['total_loss']
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Compute routing accuracy
        routing_accuracy = self._compute_routing_accuracy(batch, outputs['routing_probs'])
        
        self.global_step += 1
        
        # Prepare metrics
        metrics = {
            'total_loss': total_loss.item(),
            'routing_accuracy': routing_accuracy
        }
        
        for loss_name, loss_value in losses.items():
            if loss_name != 'total_loss':
                metrics[loss_name] = loss_value.item()
        
        return metrics
    
    def _train_single_task_batch(
        self,
        batch: Dict[str, Any],
        task_name: str,
        optimizer: optim.Optimizer,
        scheduler: Any,
        loss_fn: CurriculumLoss
    ) -> Dict[str, float]:
        """Train on a single task batch."""
        batch = move_to_device(batch, self.device)
        
        # Create task labels for oracle routing
        batch_size = batch['input_ids'].size(0)
        task_labels = torch.full(
            (batch_size,), 
            self.model.task_to_idx[task_name], 
            device=self.device
        )
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            task_labels=task_labels,
            return_routing_info=True
        )
        
        # Prepare targets
        task_targets = {task_name: batch['target']}
        
        # Compute losses (simplified for single task)
        losses = loss_fn(
            task_outputs=outputs['task_outputs'],
            task_targets=task_targets,
            routing_probs=outputs['routing_probs'],
            training_phase="phase3"
        )
        
        # Backward pass
        total_loss = losses['total_loss']
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        self.global_step += 1
        
        # Compute task-specific accuracy
        task_accuracy = self._compute_task_accuracy(
            outputs['task_outputs'].get(task_name), 
            batch['target'], 
            task_name
        )
        
        return {
            'total_loss': total_loss.item(),
            f'{task_name}_accuracy': task_accuracy
        }
    
    def _validate_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: DynamoLoss
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        loss_meters = {
            'total_loss': AverageMeter(),
            'routing_accuracy': AverageMeter()
        }
        
        task_performance = {task: AverageMeter() for task in self.model.task_names}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                batch = move_to_device(batch, self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_routing_info=True
                )
                
                # Prepare targets and compute loss
                task_targets = self._prepare_mixed_task_targets(batch)
                
                backbone_outputs = self.model.backbone(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                cls_embeddings = backbone_outputs.last_hidden_state[:, 0, :]
                
                losses = loss_fn(
                    task_outputs=outputs['task_outputs'],
                    task_targets=task_targets,
                    routing_probs=outputs['routing_probs'],
                    input_embeddings=cls_embeddings,
                    training_phase="phase3"
                )
                
                # Compute metrics
                routing_accuracy = self._compute_routing_accuracy(batch, outputs['routing_probs'])
                
                # Update meters
                batch_size = batch['input_ids'].size(0)
                loss_meters['total_loss'].update(losses['total_loss'].item(), batch_size)
                loss_meters['routing_accuracy'].update(routing_accuracy, batch_size)
                
                # Task-specific performance
                for task_name in self.model.task_names:
                    if task_name in outputs['task_outputs'] and task_name in task_targets:
                        task_acc = self._compute_task_accuracy(
                            outputs['task_outputs'][task_name],
                            task_targets[task_name],
                            task_name
                        )
                        task_performance[task_name].update(task_acc, batch_size)
        
        # Prepare validation metrics
        val_metrics = {key: meter.avg for key, meter in loss_meters.items()}
        val_metrics.update({f'{task}_accuracy': meter.avg for task, meter in task_performance.items()})
        
        return val_metrics
    
    def _prepare_mixed_task_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare task targets from mixed task batch."""
        task_targets = {}
        expected_outputs = batch['expected_outputs']
        
        for task_name in self.model.task_names:
            if task_name in expected_outputs:
                targets = expected_outputs[task_name]
                
                if isinstance(targets, list):
                    if task_name == 'qa':
                        task_targets[task_name] = torch.stack([
                            torch.tensor(t, dtype=torch.long, device=self.device) 
                            for t in targets
                        ])
                    else:
                        task_targets[task_name] = torch.tensor(
                            targets, dtype=torch.long, device=self.device
                        )
                elif isinstance(targets, torch.Tensor):
                    task_targets[task_name] = targets.to(self.device)
        
        return task_targets
    
    def _compute_routing_accuracy(
        self, 
        batch: Dict[str, Any], 
        routing_probs: torch.Tensor
    ) -> float:
        """Compute routing accuracy."""
        predicted_tasks = torch.argmax(routing_probs, dim=-1)
        task_labels = batch.get('task_labels', torch.zeros_like(predicted_tasks))
        
        if task_labels.dim() > 1:  # Multi-hot encoding
            true_tasks = torch.argmax(task_labels, dim=-1)
        else:
            true_tasks = task_labels
        
        correct = (predicted_tasks == true_tasks).float()
        return correct.mean().item()
    
    def _compute_task_accuracy(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        task_name: str
    ) -> float:
        """Compute task-specific accuracy."""
        if predictions is None or targets is None:
            return 0.0
        
        if task_name == 'sentiment':
            pred_labels = torch.argmax(predictions, dim=-1)
            correct = (pred_labels == targets).float()
            return correct.mean().item()
        
        elif task_name == 'qa':
            # For QA, we compute exact match for start and end positions
            start_preds = torch.argmax(predictions[:, :, 0], dim=-1)
            end_preds = torch.argmax(predictions[:, :, 1], dim=-1)
            
            start_correct = (start_preds == targets[:, 0]).float()
            end_correct = (end_preds == targets[:, 1]).float()
            
            exact_match = (start_correct * end_correct).mean().item()
            return exact_match
        
        else:
            # For generation tasks, use a simplified metric
            return 0.5  # Placeholder
    
    def _update_curriculum(self, epoch: int, total_epochs: int):
        """Update curriculum learning ratio."""
        progress = epoch / total_epochs
        self.current_curriculum_ratio = (
            self.curriculum_start_ratio * (1 - progress) + 
            self.curriculum_end_ratio * progress
        )
        
        logger.debug(f"Epoch {epoch}: Curriculum ratio = {self.current_curriculum_ratio:.3f}")
    
    def _create_curriculum_schedule(self, total_steps: int) -> Dict[int, float]:
        """Create curriculum schedule for loss weighting."""
        schedule = {}
        
        # Linear schedule from 0.5 to 1.0
        for step in range(0, total_steps, total_steps // 10):
            weight = 0.5 + 0.5 * (step / total_steps)
            schedule[step] = weight
        
        return schedule
    
    def _update_meters(
        self, 
        loss_meters: Dict[str, AverageMeter], 
        task_performance: Dict[str, AverageMeter], 
        metrics: Dict[str, float]
    ):
        """Update metric meters."""
        batch_size = 1  # Simplified
        
        for key, value in metrics.items():
            if key in loss_meters:
                loss_meters[key].update(value, batch_size)
            elif key.endswith('_accuracy') and key.replace('_accuracy', '') in task_performance:
                task_name = key.replace('_accuracy', '')
                task_performance[task_name].update(value, batch_size)
    
    def _setup_joint_optimizer(self, learning_rate: float) -> optim.Optimizer:
        """Setup optimizer for joint training with different learning rates."""
        # Separate parameter groups for router and adapters
        router_params = list(self.model.router.parameters())
        adapter_params = []
        
        for adapter in self.model.adapters.adapters.values():
            adapter_params.extend(list(adapter.parameters()))
        
        param_groups = [
            {
                'params': router_params,
                'lr': learning_rate * self.router_lr_scale,
                'name': 'router'
            },
            {
                'params': adapter_params,
                'lr': learning_rate * self.adapter_lr_scale,
                'name': 'adapters'
            }
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay
        )
        
        logger.info(f"Joint optimizer: Router LR = {learning_rate * self.router_lr_scale:.2e}, "
                   f"Adapter LR = {learning_rate * self.adapter_lr_scale:.2e}")
        
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
    
    def _save_joint_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """Save joint training checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase3")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'metrics': metrics,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, "best_joint_model.pt")
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved joint checkpoint to {checkpoint_path}")
    
    def _save_phase3_checkpoint(self, metrics: Dict[str, Any]):
        """Save complete Phase 3 checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase3")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save complete model
        self.model.save_model(checkpoint_dir)
        
        # Save training metrics
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.pt")
        torch.save(metrics, metrics_path)
        
        logger.info(f"Saved Phase 3 checkpoint to {checkpoint_dir}")
    
    def _log_wandb_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to Weights & Biases."""
        if not self.config.use_wandb:
            return
        
        wandb_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                wandb_metrics[f"phase3/{key}"] = value[-1]
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list) and sub_value:
                        wandb_metrics[f"phase3/{key}/{sub_key}"] = sub_value[-1]
            elif isinstance(value, (int, float)):
                wandb_metrics[f"phase3/{key}"] = value
        
                        if wandb.run is not None:
                    wandb.log(wandb_metrics, step=self.global_step)


def run_phase3_training(config: Config, model: DynamoModel) -> Dict[str, Any]:
    """
    Run Phase 3 training.
    
    Args:
        config: Training configuration
        model: DYNAMO model
    
    Returns:
        Training metrics
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize trainer
    trainer = Phase3Trainer(config, model)
    
    # Run training
    metrics = trainer.train_with_curriculum()
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from utils.config import get_config
    from model import DynamoModel
    
    # Load configuration
    config = get_config()
    
    # Initialize model
    model = DynamoModel(config.__dict__)
    
    # Run Phase 3 training
    metrics = run_phase3_training(config, model)
    
    print("Phase 3 training completed!")
    print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    if 'routing_accuracy' in metrics and metrics['routing_accuracy']:
        print(f"Final routing accuracy: {metrics['routing_accuracy'][-1]:.3f}")

