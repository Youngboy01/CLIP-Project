import os
import torch
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm.auto import tqdm
import warnings
from config import Config
from DataLoad import ROCODataset

config = Config()
warnings.filterwarnings("ignore", category=FutureWarning)
wandb.login()


class CLIPFineTuner:
    def __init__(self, model, strategy: str):
        self.model = model
        self.strategy = strategy
        self.apply_strategy()

    def freeze_all(self):
        """Freeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = False

    def get_layer_info(self):
        """Get information about model layers"""
        vision_layers = []
        text_layers = []

        if hasattr(self.model.vision_model, "encoder"):
            vision_layers = list(self.model.vision_model.encoder.layers)
        if hasattr(self.model.text_model, "encoder"):
            text_layers = list(self.model.text_model.encoder.layers)

        return vision_layers, text_layers

    def apply_strategy(self):
        print(f"\n{'=' * 60}")
        print(f"Applying Fine-tuning Strategy: {self.strategy.upper()}")
        print(f"{'=' * 60}\n")

        self.freeze_all()

        if self.strategy == "vision_only":
            self._apply_vision_only()
        elif self.strategy == "text_only":
            self._apply_text_only()
        elif self.strategy == "last_30":
            self._apply_last_30()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._print_trainable_params()

    def _apply_vision_only(self):
        print("Unfreezing: Vision Encoder")
        for param in self.model.vision_model.parameters():
            param.requires_grad = True

        if hasattr(self.model, "visual_projection"):
            for param in self.model.visual_projection.parameters():
                param.requires_grad = True

    def _apply_text_only(self):
        print("Unfreezing: Text Encoder")
        for param in self.model.text_model.parameters():
            param.requires_grad = True

        if hasattr(self.model, "text_projection"):
            for param in self.model.text_projection.parameters():
                param.requires_grad = True

    def _apply_last_30(self):
        """Unfreeze last 30% of layers in both encoders"""
        vision_layers, text_layers = self.get_layer_info()

        vision_threshold = int(len(vision_layers) * 0.7)
        text_threshold = int(len(text_layers) * 0.7)

        print(f"Vision Encoder: {len(vision_layers)} layers total")
        print(f"  - Freezing first {vision_threshold} layers")
        print(f"  - Unfreezing last {len(vision_layers) - vision_threshold} layers")

        print(f"\nText Encoder: {len(text_layers)} layers total")
        print(f"  - Freezing first {text_threshold} layers")
        print(f"  - Unfreezing last {len(text_layers) - text_threshold} layers")

        for i in range(vision_threshold, len(vision_layers)):
            for param in vision_layers[i].parameters():
                param.requires_grad = True

        if hasattr(self.model.vision_model, "post_layernorm"):
            print("  - Unfreezing Vision Post-LayerNorm")
            for param in self.model.vision_model.post_layernorm.parameters():
                param.requires_grad = True

        for i in range(text_threshold, len(text_layers)):
            for param in text_layers[i].parameters():
                param.requires_grad = True

        if hasattr(self.model.text_model, "final_layer_norm"):
            print("  - Unfreezing Text Final-LayerNorm")
            for param in self.model.text_model.final_layer_norm.parameters():
                param.requires_grad = True

        if hasattr(self.model, "visual_projection"):
            for param in self.model.visual_projection.parameters():
                param.requires_grad = True

        if hasattr(self.model, "text_projection"):
            for param in self.model.text_projection.parameters():
                param.requires_grad = True

    def _print_trainable_params(self):
        """Print summary of trainable parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        percentage = 100 * trainable / total

        print(f"\n{'=' * 60}")
        print(f"Trainable Parameters: {trainable:,} / {total:,} ({percentage:.2f}%)")
        print(f"{'=' * 60}\n")


class CLIPTrainer:
    """Trainer for CLIP fine-tuning"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        num_update_steps_per_epoch = (
            len(train_loader) // config.gradient_accumulation_steps
        )
        max_train_steps = config.num_epochs * num_update_steps_per_epoch

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=max_train_steps,
        )
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        self.global_step = 0
        self.best_val_loss = float("inf")

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_grad_norm = 0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
        )

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(self.config.device)
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_loss=True,
                    )

                    loss = outputs.loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True,
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            current_loss = loss.item() * self.config.gradient_accumulation_steps
            total_loss += current_loss

            if (step + 1) % self.config.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(self.train_loader):
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                if (
                    self.config.use_wandb
                    and self.global_step % self.config.log_every_n_steps == 0
                ):
                    wandb.log(
                        {
                            "train/loss": current_loss,
                            "train/grad_norm": grad_norm.item(),
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "global_step": self.global_step,
                        }
                    )
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate with Recall@1 and Recall@5"""
        self.model.eval()
        total_loss = 0
        total_correct_r1 = 0
        total_correct_r5 = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(self.config.device)
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True,
                )
            logits = outputs.logits_per_image
            batch_size = logits.shape[0]
            labels = torch.arange(batch_size, device=self.config.device)

            pred = logits.argmax(dim=1)
            total_correct_r1 += (pred == labels).sum().item()

            if batch_size >= 5:
                _, top5_indices = logits.topk(5, dim=1)
                total_correct_r5 += (
                    (top5_indices == labels.view(-1, 1)).any(dim=1).sum().item()
                )
            else:
                total_correct_r5 += (pred == labels).sum().item()

            total_loss += outputs.loss.item()
            total_samples += batch_size
            avg_loss = total_loss / len(self.val_loader)
            avg_r1 = total_correct_r1 / total_samples
            avg_r5 = total_correct_r5 / total_samples

            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "r1": f"{avg_r1:.4f}",
                    "r5": f"{avg_r5:.4f}",
                }
            )

        return avg_loss, avg_r1, avg_r5

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
            "config": vars(self.config),
        }
        path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        if is_best:
            best_pt_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_pt_path)
            hf_save_dir = os.path.join(self.config.checkpoint_dir, "best_model_hf")
            self.model.save_pretrained(hf_save_dir)
            print(f"Saved best model (HF Ready): {hf_save_dir}")

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60 + "\n")

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch)

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")

            val_loss = None
            is_best = False

            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_loss, val_r1, val_r5 = self.validate()

                print(
                    f"Val Loss: {val_loss:.4f} | R@1: {val_r1:.4f} | R@5: {val_r5:.4f}"
                )

                if self.config.use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/recall_1": val_r1,
                            "val/recall_5": val_r5,
                            "epoch": epoch + 1,
                        }
                    )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = True
                    print("New best model found!")

            should_save = (
                (epoch + 1) % self.config.save_every_n_epochs == 0
            ) or is_best

            if should_save:
                save_loss = val_loss if val_loss is not None else self.best_val_loss
                self.save_checkpoint(epoch, save_loss, is_best)

            print()

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint to resume training"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["val_loss"]

        print(f"Resuming from Epoch {start_epoch}")
        return start_epoch
