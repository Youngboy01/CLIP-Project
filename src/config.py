import torch


class Config:
    model_name = "openai/clip-vit-base-patch32"
    dataset_name = "eltorio/ROCO-radiology"
    max_length = 77
    image_size = 224

    strategy: Literal["vision_only", "text_only", "last_30"] = "last_30"
    num_epochs = 5
    batch_size = 128
    actual_batch_size = 32
    gradient_accumulation_steps = 4
    learning_rate = 1e-5
    warmup_steps = 500
    weight_decay = 0.01
    max_grad_norm = 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True
    num_workers = 2

    use_wandb = True
    checkpoint_dir = "./checkpoints"
    save_every_n_epochs = 1
    log_every_n_steps = 50

    eval_every_n_epochs = 1
