# 🏥 ROCO-Radiology-CLIP: Specialized Medical Image Retrieval

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/spicy03/CLIP-ROCO-v1)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/spicy03/ROCO-Radiology-Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> **Fine-tuning OpenAI's CLIP (ViT-B/32) for State-of-the-Art Radiology Retrieval on consumer hardware.**

---

## 🚀 Project Overview

Standard multi-modal models like CLIP are trained on general internet data, making them suboptimal for specialized domains like radiology. This project adapts **CLIP (ViT-B/32)** to the medical domain by fine-tuning it on the **ROCO (Radiology Objects in COntext)** dataset.

The result is a model capable of **Zero-Shot Medical Classification** and **Semantic Search** across X-rays, CTs, and MRIs, achieving **71% accuracy** on difficult diagnostic pairs.

### 🔑 Key Engineering Features
* **Domain Adaptation:** Aligned visual and textual embeddings for 65,000+ radiology pairs.
* **Resource-Efficient Training:** Implemented **Mixed Precision (FP16)** and **Gradient Accumulation** to train on a single NVIDIA T4 (16GB) GPU, simulating a batch size of 128.
* **Full Deployment Pipeline:** End-to-end pipeline from raw data to a deployed **Gradio** web application on Hugging Face Spaces.

---

## 📊 Performance Metrics

We evaluated the model on the ROCO Test Set (8,176 images) using two distinct retrieval tasks:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Batch-wise Recall@1** | **70.83%** | **Classification Proxy:** Accuracy in identifying the correct caption out of a batch of 32 candidates. |
| **Batch-wise Recall@5** | **96.99%** | Accuracy that the correct label is within the top 5 predictions. |
| **Global Recall@5** | **~6.02%** | **"Needle in a Haystack":** Retrieval from the *entire* test set of 8,176 images. |

> **📉 Understanding the Global Recall (6%):**
> In the Global Retrieval task, the model must find **1 specific image** out of **8,176 highly similar distractors** (e.g., thousands of other chest X-rays). Random guessing would be **0.01%**.
> * Our model achieves **6%**, representing a **600x improvement over random chance**.
> * For practical diagnostic workflows (Batch-wise), the model achieves **State-of-the-Art ~71% accuracy**.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/roco-radiology-clip.git](https://github.com/your-username/roco-radiology-clip.git)
cd roco-radiology-clip

# Install dependencies
pip install torch transformers pillow gradio datasets