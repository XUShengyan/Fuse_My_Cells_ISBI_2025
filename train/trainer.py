import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from train.loss_manager import LossManager, SSIM3D, NSSIMLoss
from train.n_ssim import calculate_n_ssim

class Trainer:
    """Trainer class for managing model training, validation, and logging."""

    def __init__(self, model, train_loader, val_loader, config):
        """Initialize the trainer with model, data loaders, and configuration."""
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config.training
        self.config_all = config

        # Initialize SSIM metric calculator
        ssim_params = {}
        for cfg in config.loss_configs:
            if cfg['loss_name'] == 'ssim3d':
                ssim_params = cfg.get('params', {})
                break
        self.ssim_metric = calculate_n_ssim

        # Basic settings
        weight_decay = self.config.weight_decay
        self.use_amp = self.config.use_amp
        self.use_amp_valid = False
        self.device = self.config.device
        self.log_dir = config.paths.log_dir
        self.save_model_dir = config.paths.save_model_dir

        # Optimizer
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self._grad_clip_max_norm = 0.1
        lr = float(self.config.lr)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4 if weight_decay == 0 else weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.scheduler = self._build_scheduler()

        # Loss manager
        self.loss_manager = self._build_loss_manager(config.loss_configs)

        # Mixed precision training
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        # Training state
        self.best_val_loss = float('inf')
        self.best_val_NSSIM = float('-inf')
        self.current_epoch = 0

        # Logging and saving
        self.writer = SummaryWriter(log_dir=self.log_dir)
        os.makedirs(self.save_model_dir, exist_ok=True)

        # Initialize log file
        log_file_path = os.path.join(self.save_model_dir, 'training_log.txt')
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Training Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        accumulation_steps = 0
        epoch_loss = 0.0
        total_n_ssim = 0.0
        total_pred_ssim = 0.0
        total_batch_num = len(self.train_loader)

        auto_dtype = torch.float if not self.use_amp else torch.float16

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True, dtype=auto_dtype)
            targets = targets.to(self.device, non_blocking=True, dtype=auto_dtype)

            outputs = self.model(inputs)
            loss, loss_details = self.loss_manager.compute_loss_with_details(outputs, targets, inputs)
            epoch_loss += loss.item()

            loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_steps += 1

            if accumulation_steps % self.gradient_accumulation_steps == 0:
                self._update_parameters()
                accumulation_steps = 0

            with torch.no_grad():
                n_ssim_batch, pred_ssim = self.ssim_metric(inputs, outputs, targets)
                total_n_ssim += n_ssim_batch.item()
                total_pred_ssim += pred_ssim.item()

            for name, value in loss_details.items():
                self.writer.add_scalar(f'Loss/train_{name}', value, self.current_epoch)

        if accumulation_steps > 0:
            self._update_parameters()

        train_loss = epoch_loss / total_batch_num
        train_n_ssim = total_n_ssim / total_batch_num
        train_pred_ssim = total_pred_ssim / total_batch_num

        return train_loss, train_n_ssim, train_pred_ssim

    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        total_n_ssim = 0.0
        total_pred_ssim = 0.0
        total_batch_num = len(self.val_loader)

        auto_dtype = torch.float if not self.use_amp_valid else torch.float16

        with torch.inference_mode():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True, dtype=auto_dtype)
                targets = targets.to(self.device, non_blocking=True, dtype=auto_dtype)

                outputs = self.model(inputs)
                loss, loss_details = self.loss_manager.compute_loss_with_details(outputs, targets, inputs)
                total_loss += loss.item()

                n_ssim_batch, pred_ssim = self.ssim_metric(inputs, outputs, targets)
                total_n_ssim += n_ssim_batch.item()
                total_pred_ssim += pred_ssim.item()

        val_loss = total_loss / total_batch_num
        val_n_ssim = total_n_ssim / total_batch_num
        val_pred_ssim = total_pred_ssim / total_batch_num

        return val_loss, val_n_ssim, val_pred_ssim

    def save_checkpoint(self, is_best=False):
        """Save the model checkpoint."""
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state, os.path.join(self.save_model_dir, "latest_checkpoint.pth"))
        if is_best:
            torch.save(state, os.path.join(self.save_model_dir, "best_model.pth"))

    def load_checkpoint(self, checkpoint_path):
        """Load a model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def train_model(self):
        """Train the model for the specified number of epochs."""
        total_epochs = self.config.epochs
        log_file_path = os.path.join(self.save_model_dir, 'training_log.txt')
        self._log_initial_training_info()

        for epoch in range(total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_loss, train_n_ssim, train_pred_ssim = self.train_epoch()
            val_loss, val_n_ssim, val_pred_ssim = self.validate()

            self._update_scheduler(val_loss)

            is_best = val_n_ssim > self.best_val_NSSIM
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_NSSIM = val_n_ssim

            if ((epoch + 1) % self.config.save_interval == 0) or (epoch + 1 == total_epochs) or is_best:
                self.save_checkpoint(is_best=is_best)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('N_SSIM/train', train_n_ssim, epoch)
            if val_loss is not None:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('N_SSIM/val', val_n_ssim, epoch)

            epoch_duration = time.time() - epoch_start_time
            remaining_epochs = total_epochs - (epoch + 1)
            estimated_remaining = epoch_duration * remaining_epochs

            def format_time(seconds):
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

            log_lines = [
                f"Epoch: {epoch + 1}/{total_epochs}",
                f"Train Loss: {train_loss:.4e}, N_SSIM: {train_n_ssim:.4f}, Pred_SSIM: {train_pred_ssim:.4f}",
            ]
            if val_loss is not None:
                log_lines.append(
                    f"Val Loss: {val_loss:.4e}, Val N_SSIM: {val_n_ssim:.4f}, Pred_SSIM: {val_pred_ssim:.4f}")
            log_lines.append(
                f"Time: Epoch Duration: {format_time(epoch_duration)}, Estimated Remaining: {format_time(estimated_remaining)}")
            if is_best:
                log_lines.append(f"New best N_SSIM: {val_n_ssim:.4f}")

            log_str = '\n'.join(log_lines)
            print(log_str + '\n')

            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_str + '\n\n')

    def finalize(self):
        """Finalize training by closing the TensorBoard writer."""
        self.writer.close()
        print(f"Training completed with best validation loss: {self.best_val_loss:.4e}")

    def _build_loss_manager(self, loss_configs: list) -> LossManager:
        """Build the loss manager based on configuration."""
        loss_mgr = LossManager([], self.device)
        loss_registry = {
            'mse': lambda **params: nn.MSELoss(**params),
            'l1': lambda **params: nn.L1Loss(**params),
            'ssim3d': lambda **params: SSIM3D(**params),
            'nssim': lambda **params: NSSIMLoss(**params)
        }
        for cfg in loss_configs:
            name = cfg['loss_name']
            if name not in loss_registry:
                raise ValueError(f"Unknown loss function: {name}")
            loss_instance = loss_registry[name](**cfg.get('params', {}))
            loss_mgr.add_loss(
                name=name,
                weight=cfg['weight'],
                impl=loss_instance.to(self.device),
                params={}
            )
        return loss_mgr

    def _build_scheduler(self):
        """Build the learning rate scheduler based on configuration."""
        cfg = self.config.scheduler
        scheduler_type = cfg['name'].lower()
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(cfg.get('plateau_factor', 0.2)),
                patience=int(cfg.get('plateau_patience', 5)),
                min_lr=float(cfg.get('min_lr', 1e-5))
            )
        elif scheduler_type == "cosine_with_warmup":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return self._create_cosine_with_warmup(
                total_epochs=self.config.epochs,
                warmup_epochs=int(cfg.get('warmup_epochs', 5)),
                min_lr=float(cfg.get('min_lr', 1e-6)),
                cycles=int(cfg.get('decay_cycles', 1))
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(cfg['step_size']),
                gamma=float(cfg['step_gamma'])
            )
        elif scheduler_type == "x":
            return None
        elif scheduler_type == "cosine_restart":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=15,
                T_mult=1,
                eta_min=1e-5
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _create_cosine_with_warmup(self, total_epochs, warmup_epochs, min_lr, cycles=1):
        """Create a custom cosine annealing scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (1 - min_lr) * (1 + np.cos(np.pi * progress * cycles))
        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _update_scheduler(self, val_loss):
        """Update the learning rate scheduler."""
        if self.scheduler is not None:
            cfg = self.config.scheduler
            if cfg['name'] == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def _log_initial_training_info(self):
        """Log initial training information to a file."""
        log_file_path = os.path.join(self.save_model_dir, 'training_log.txt')
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write("====== Training Initialization Information ======\n")
            f.write(f"Training Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Model information
            f.write("【Model Information】\n")
            f.write(f"Model Name: {self.config_all.model.name}\n")
            f.write(f"Model Class: {self.model.__class__.__name__}\n")
            try:
                model_params = self.config_all.model.params
                f.write("Model Parameters:\n")
                for key, value in model_params.items():
                    f.write(f"\t{key}: {value}\n")
            except AttributeError:
                f.write("Model Parameters: Not available\n")
            f.write("\n")

            # Optimizer information
            f.write("【Optimizer Information】\n")
            f.write(f"Optimizer Type: {self.optimizer.__class__.__name__}\n")
            for idx, param_group in enumerate(self.optimizer.param_groups):
                f.write(f"Parameter Group {idx}:\n")
                f.write(f"\tLearning Rate: {param_group.get('lr', 'N/A')}\n")
                f.write(f"\tWeight Decay: {param_group.get('weight_decay', 'N/A')}\n")
                f.write(f"\tBetas: {param_group.get('betas', 'N/A')}\n")
                f.write(f"\tEps: {param_group.get('eps', 'N/A')}\n")
            f.write("\n")

            # Scheduler information
            f.write("【Scheduler Information】\n")
            if self.scheduler is not None:
                scheduler_name = self.config.scheduler['name']
                f.write(f"Scheduler Type: {scheduler_name}\n")
                scheduler_name_lower = scheduler_name.lower()
                if scheduler_name_lower == "plateau":
                    f.write(f"\tPlateau Factor: {self.config.scheduler.get('plateau_factor', 'N/A')}\n")
                    f.write(f"\tPlateau Patience: {self.config.scheduler.get('plateau_patience', 'N/A')}\n")
                    f.write(f"\tMin LR: {self.config.scheduler.get('min_lr', 'N/A')}\n")
                elif scheduler_name_lower == "cosine_with_warmup":
                    f.write(f"\tWarmup Epochs: {self.config.scheduler.get('warmup_epochs', 'N/A')}\n")
                    f.write(f"\tMin LR: {self.config.scheduler.get('min_lr', 'N/A')}\n")
                    f.write(f"\tDecay Cycles: {self.config.scheduler.get('decay_cycles', 'N/A')}\n")
                elif scheduler_name_lower == "step":
                    f.write(f"\tStep Size: {self.config.scheduler.get('step_size', 'N/A')}\n")
                    f.write(f"\tStep Gamma: {self.config.scheduler.get('step_gamma', 'N/A')}\n")
                elif scheduler_name_lower == "cosine_restart":
                    f.write("\tUsing CosineAnnealingWarmRestarts Scheduler\n")
                else:
                    f.write("\tScheduler configuration details not defined\n")
            else:
                f.write("No Scheduler\n")
            f.write("\n")

            # Training parameters
            f.write("【Training Parameters】\n")
            f.write(f"Total Epochs: {self.config.epochs}\n")
            f.write(f"Batch Size (Training): {self.train_loader.batch_size}\n")
            f.write(f"Batch Size (Validation): {self.val_loader.batch_size}\n")
            try:
                training_params = self.config_all.training
                f.write("YAML Parameters:\n")
                f.write(f"Train Batches per Epoch: {training_params.train_batches_per_epoch}\n")
                f.write(f"Val Batches per Epoch: {training_params.val_batches_per_epoch}\n")
                f.write(f"Gradient Accumulation Steps: {training_params.gradient_accumulation_steps}\n")
            except AttributeError:
                f.write("YAML Parameters: Not available\n")
            f.write("\n")

            # Loss functions
            f.write("【Loss Functions】\n")
            for name, cfg in self.loss_manager.loss_funcs.items():
                f.write(f"Loss Function: {name}  Weight: {cfg['weight']}\n")
            f.write("====== Start Training ======\n\n")

    def _update_parameters(self):
        """Unified parameter update logic with gradient clipping and AMP handling."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip_max_norm)
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

