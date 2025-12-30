import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import json
from datetime import datetime
import os


@torch.no_grad()
def evaluate_model():
    print("-" * 50)
    print("testing on the test dataset")
    print("-" * 50)
    MODEL_PATH = "/content/drive/MyDrive/best_model_hf"
    print(f"loading model from path : {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print(f"Model successfully loaded for inferencing on device : {device}")
    print("\n Load the test dataset")
    test_dataset = load_dataset("eltorio/ROCO-radiology", split="test")
    print("\n test data loaded")
    batch_size = 64
    all_image_features = []
    all_text_features = []
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch_end = min(i + batch_size, len(test_dataset))
        batch = test_dataset[i:batch_end]

        images = [item.convert("RGB") for item in batch["image"]]
        captions = batch["caption"]

        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(
            device
        )
        text_inputs = processor(
            text=captions, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        all_image_features.append(F.normalize(image_features, dim=-1))
        all_text_features.append(F.normalize(text_features, dim=-1))
    final_image_features = torch.cat(all_image_features, dim=0)
    final_text_features = torch.cat(all_text_features, dim=0)

    print("all features encoded")

    print("calculating similarity scores")
    similarity_scores = torch.matmul(final_image_features, final_text_features.t())
    print("similarity scores calculated")

    print("calculating retrieval metrics")
    n = similarity_scores.shape[0]

    img2txt_ranks = []
    for i in tqdm(range(n), desc="Image→Text"):
        scores = similarity_scores[i]
        rank = (scores.argsort(descending=True) == i).nonzero(as_tuple=True)[0].item()
        img2txt_ranks.append(rank)

    txt2img_ranks = []
    for i in tqdm(range(n), desc="Text→Image"):
        scores = similarity_scores[:, i]
        rank = (scores.argsort(descending=True) == i).nonzero(as_tuple=True)[0].item()
        txt2img_ranks.append(rank)
    img2txt_ranks = np.array(img2txt_ranks)
    txt2img_ranks = np.array(txt2img_ranks)
    metrics = {
        "model_path": MODEL_PATH,
        "test_samples": n,
        "evaluation_date": datetime.now().isoformat(),
        "image_to_text": {
            "recall_at_1": float((img2txt_ranks < 1).mean() * 100),
            "recall_at_5": float((img2txt_ranks < 5).mean() * 100),
            "recall_at_10": float((img2txt_ranks < 10).mean() * 100),
            "median_rank": float(np.median(img2txt_ranks) + 1),
            "mean_rank": float(np.mean(img2txt_ranks) + 1),
        },
        "text_to_image": {
            "recall_at_1": float((txt2img_ranks < 1).mean() * 100),
            "recall_at_5": float((txt2img_ranks < 5).mean() * 100),
            "recall_at_10": float((txt2img_ranks < 10).mean() * 100),
            "median_rank": float(np.median(txt2img_ranks) + 1),
            "mean_rank": float(np.mean(txt2img_ranks) + 1),
        },
    }
    metrics["average_recall_at_5"] = (
        metrics["image_to_text"]["recall_at_5"]
        + metrics["text_to_image"]["recall_at_5"]
    ) / 2

    print("\n" + "=" * 70)
    print(" FINAL TEST SET RESULTS")
    print("=" * 70)

    print("\nImage → Text Retrieval:")
    print(f"  R@1:  {metrics['image_to_text']['recall_at_1']:.2f}%")
    print(f"  R@5:  {metrics['image_to_text']['recall_at_5']:.2f}%")
    print(f"  R@10: {metrics['image_to_text']['recall_at_10']:.2f}%")
    print(f"  Median Rank: {metrics['image_to_text']['median_rank']:.1f}")

    print("\nText → Image Retrieval:")
    print(f"  R@1:  {metrics['text_to_image']['recall_at_1']:.2f}%")
    print(f"  R@5:  {metrics['text_to_image']['recall_at_5']:.2f}%")
    print(f"  R@10: {metrics['text_to_image']['recall_at_10']:.2f}%")
    print(f"  Median Rank: {metrics['text_to_image']['median_rank']:.1f}")

    print(f"\n Average R@5: {metrics['average_recall_at_5']:.2f}%")

    avg_r5 = metrics["average_recall_at_5"]
    if avg_r5 > 70:
        print(" super performance")
    elif avg_r5 > 60:
        print(" good performance!.")
    elif avg_r5 > 50:
        print("  fair performance. Review needed")
    else:
        print("  poor performance. retrain.")

    drive_dir = "/content/drive/MyDrive/best_model_hf"
    results_file = os.path.join(drive_dir, "evaluation_results.json")

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n Metrics saved safely to: {results_file}")
    return metrics
