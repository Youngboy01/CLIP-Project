import os
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from config import Config
from DataLoad import ROCODataset
from clipTrainer import CLIPFineTuner,CLIPTrainer
config = Config()


def main():
    DEBUG = False
    RESUME_FROM = None

    drive_checkpoint_dir = "/content/drive/MyDrive/clip_roco_finetuning"
    config.checkpoint_dir = drive_checkpoint_dir
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print(f"CLIP Fine-tuning on ROCO (Production Mode)")
    print(f"Checkpoints will save to: {config.checkpoint_dir}")
    print("=" * 60)

    if config.use_wandb:
        run_name = f"clip_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if DEBUG:
            run_name += "_DEBUG"

        wandb.init(
            project="clip-roco-finetuning",
            config=vars(config),
            name=run_name,
            resume="allow",
        )

    print("Loading CLIP...")
    processor = CLIPProcessor.from_pretrained(config.model_name)
    model = CLIPModel.from_pretrained(config.model_name)

    CLIPFineTuner(model, config.strategy)

    print("Loading Datasets...")
    train_limit = 100 if DEBUG else None
    val_limit = 50 if DEBUG else None

    train_dataset = ROCODataset(
        split="train", processor=processor, max_samples=train_limit
    )
    val_dataset = ROCODataset(
        split="validation", processor=processor, max_samples=val_limit
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.actual_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.actual_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    trainer = CLIPTrainer(model, train_loader, val_loader, config)

    start_epoch = 0

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming from: {RESUME_FROM}")
        start_epoch = trainer.load_checkpoint(RESUME_FROM)

    print(f"Starting Training from Epoch {start_epoch + 1}...")

    try:
        for epoch in range(start_epoch, config.num_epochs):
            train_loss = trainer.train_epoch(epoch)
            print(f"\nEpoch {epoch + 1}/{config.num_epochs} | Loss: {train_loss:.4f}")

            if (epoch + 1) % config.eval_every_n_epochs == 0:
                val_loss, val_r1, val_r5 = trainer.validate()
                print(
                    f"Val Loss: {val_loss:.4f} | R@1: {val_r1:.4f} | R@5: {val_r5:.4f}"
                )

                if config.use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/recall_1": val_r1,
                            "val/recall_5": val_r5,
                            "epoch": epoch + 1,
                        }
                    )

                is_best = val_loss < trainer.best_val_loss
                if is_best:
                    trainer.best_val_loss = val_loss
                    print("New best model found!")

                if ((epoch + 1) % config.save_every_n_epochs == 0) or is_best:
                    trainer.save_checkpoint(epoch, val_loss, is_best)

            print()

        print("\n Full Training Completed Successfully!")

    except KeyboardInterrupt:
        print("\n Training interrupted by user.")
    except Exception as e:
        print(f"\n Error during training: {e}")
        raise
    finally:
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
