#  ROCO-Radiology-CLIP: Specialized Medical Image Retrieval

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/spicy03/CLIP-ROCO-v1)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/spicy03/ROCO-Radiology-Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> **Fine-tuning OpenAI's CLIP for Radiology Retrieval.**

---

##  Project Overview
This project adapts **CLIP (Contrastive Language–Image Pretraining)** to the medical imaging domain by fine-tuning it on the **ROCO (Radiology Objects in COntext)** dataset.  
The goal is to improve **cross-modal retrieval** between radiology images and their corresponding clinical captions.

----

## Training Setup (Version 1)
### Base Model
- CLIP (ViT-B/32)

### Batch Configuration
- Actual batch size: **32**
- Gradient accumulation steps: **4**
- Effective batch size: **128**

### Fine-tuning Strategy
- Only the **last 30% of layers** in both the **vision encoder** and **text encoder** were unfrozen.
- I made this choice as I thought that **higher layers capture domain-specific semantic alignment**, while lower layers transfer well from natural-image pretraining.

Additional configuration details can be found in `src/config.py`.

---

## Global Retireval meaning
Before we see the metrics i want to explain what is meant by global retrieval means.

**Global retrieval** means that for every query (image or text), the model retrieves results by ranking against the **entire candidate pool**.

In this project:
- Each query is ranked against **all 8,176 images (or their paired captions)**.
- There is **no restriction** such as batch-level, patient-level, or subset-level retrieval.
- Every query competes with **every other sample** in the dataset.

##  Performance Metrics for v1.

###  Evaluation Setup

The model was evaluated on the **ROCO test set (8,176 image–caption pairs)** using two retrieval protocols:
1. **Batch-wise retrieval**
2. **Global retrieval**

---

###  Batch-wise Retrieval Metrics
*(Batch size = 32; primarily a training-time diagnostic)*

| Metric | Score | Description |
|------|------|-------------|
| **Batch-wise Recall@1** | **70.83%** | Correct caption ranked first among 32 candidates |
| **Batch-wise Recall@5** | **96.99%** | Correct caption appears within the top 5 predictions |

Batch-wise recall acts as a **classification proxy** and reflects how well the model separates positive and negative pairs within a small candidate set.

---

##  Global Retrieval Metrics
*(Retrieval over the full dataset)*

| Metric | Score | Description |
|------|------|-------------|
| **Global Recall@1** | **~6.02%** | Correct image–text pair ranked first among all 8,176 candidates |
---

##  What Do These Metrics Mean?

### Recall@1
- Measures the fraction of queries where the **top-ranked result** is the correct match.
- A strict metric that reflects **exact cross-modal alignment**.

### Recall@5
- Measures whether the correct match appears within the **top 5 retrieved results**.
- Indicates whether the model places the correct item **near the top**, even if not at rank 1.

### Batch-wise vs Global Recall
- Batch-wise recall is **easier** and often optimistic.
- Global recall is **significantly harder** and better reflects **real-world retrieval performance**.

---
## Understanding the Global Recall Result (~6%)

In the global retrieval setting:
- The model must retrieve **one correct image** from **8,176 highly similar candidates**  
  (e.g., many chest X-rays with overlapping clinical descriptions).
- A random model would achieve approximately **0.01% Recall@1**.
- This model achieves **~6% Recall@1**, which is **several hundred times better than random**.

Although the absolute value may appear modest, this result demonstrates **meaningful cross-modal semantic alignment**, given:
- the fine-grained nature of radiology language,
- strong visual similarity across images,
- and the fact that CLIP was originally trained on natural images rather than medical data.

I am trying to improve this value.

## **Then i tried changing a few things in v2**
---

###  Architectural Change

- **Model upgrade:**  
  - From **CLIP (ViT-B/32)** → **CLIP (ViT-B/16)**  
- **Motivation:**  
  - ViT-B/16 operates on **smaller patch sizes**, enabling the vision encoder to capture **finer spatial details**, which i thought my model needs since these images have intricate details which may be missed by larger patches.
---
### Batch Size and Optimization Strategy

- **Gradient accumulation steps:** increased from **4 → 8**
- **Actual batch size:** kept at **32**
- **Effective batch size:** increased from **128 → 256**

**Reasoning:**
- Larger batch sizes may stabilize **contrastive learning** and improve representation alignment in CLIP-style training.
- I avoided increasing the actual batch size to 64 due to concerns about **GPU memory limitations**.
- Instead, I opted for higher gradient accumulation to simulate a larger batch.

> In hindsight, experimenting with a larger *actual* batch size (e.g., 64) may be worthwhile, since right now clip model just check in the batch of 32 images. So 1 image compared against 31 other. This does not give it enough negative samples. So increasing to 64 may help out.

---
###  Data Augmentation

- **Added image augmentations** to the training pipeline.
- **Motivation:**
  - Since the actual batch size was not increased, I thought augmenting the images may give **additional variability**.
  - This helps reduce overfitting and encourages the model to learn more **robust visual representations**, which can be important in a dataset like ROCO.

---
###  Training Dynamics Observed

- The **training loss curve became smoother** compared to Version 1.
- The **best-performing checkpoint** was obtained at **epoch 5**, compared to **epoch 3** in Version 1.
- This suggests:
  - More stable optimization
  - Better utilization of training data
  - Reduced early overfitting

---
##  Performance Metrics for Version 2

The model was evaluated on the **ROCO test set (8,176 image–caption pairs)** using both batch-wise and global retrieval protocols.

| Metric | Score | Description |
|------|------|-------------|
| **Batch-wise Recall@1** | **72.59%** | Correct caption ranked first among 32 candidates |
| **Batch-wise Recall@5** | **97.92%** | Correct caption appears within the top 5 predictions |
| **Global Recall@1** | **~8.41%** | Correct match ranked first among all 8,176 candidates |
| **Global Recall@5** | **~19.39%** | Correct match appears within the top 5 across the entire dataset |

---
###  Summary of Improvements from v1 → v2

- Improved **batch-wise Recall@1** (70.83% → 72.59%)
- Noticeable gains in **global retrieval**:
  - Recall@1: ~6.02% → ~8.41%
  - Recall@5: ~16% → ~19.39%
- Better training stability and later optimal convergence

These results indicate that **higher visual resolution, larger effective batch size, and data augmentation** collectively contributed to stronger global retrieval performance.

---
I am continuing to iterate on these design choices to further improve global recall, particularly by exploring:
-  larger batch sizes(the actual ones),
- more aggressive hard-negative mining since there are lesser number of negatives right now,
- and alternative fine-tuning strategies like freezing just the text encoder or just the vision encoder.

