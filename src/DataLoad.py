from torch.utils.data import Dataset
from datasets import load_dataset
from config import Config

config = Config()


class ROCODataset(Dataset):
    """ROCO Radiology Dataset with CLIP preprocessing"""

    def __init__(self, split="train", processor=None, max_samples=None):
        self.processor = processor
        self.split = split  # Added this line

        print(f"Loading ROCO dataset ({split} split)...")
        dataset = load_dataset(config.dataset_name, split=split)

        print("Filtering invalid samples...")
        original_len = len(dataset)

        dataset = dataset.filter(
            lambda x: x["image"] is not None
            and x["caption"] is not None
            and len(x["caption"].strip()) > 0
        )

        print(f"Filtered {original_len - len(dataset)} invalid samples.")

        if max_samples:
            print(f"Selecting first {max_samples} samples for debug...")
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.data = dataset
        print(f"Final dataset size: {len(self.data)} samples")
        self.augment = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.split == "train":
            image = self.augment(image)

        caption = item["caption"]

        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=config.max_length,
            truncation=True,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
