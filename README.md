# Intent Classification with BERT (Pi School Project)

**Note:** This repository contains a subset of the code developed for a project at **Pi School**. The purpose of sharing this is to demonstrate my work for the admission selection committee. Certain supporting files and datasets are **confidential** and therefore not included. The provided code and this README are sufficient to understand the methodology, training, and evaluation of the project.

---

## Overview

Effective communication between users and digital systems relies on correctly identifying user intentions expressed in natural language. This project develops a robust **Intent Classification system for Italian language utterances** using state-of-the-art NLP techniques.  

Key features of the project:

- Fine-tuned **Italian BERT model** for supervised intent classification.
- **Confidence-calibrated predictions** for operational reliability.
- **Text normalization pipeline** for consistent downstream processing.
- **Intent Discovery mechanism** to detect new or ambiguous intents.

This system was designed to support a B2B voicebot, enabling automation while maintaining high accuracy in real-world conditions.

---

## Data

**Training Data:**

- 3,000+ manually labeled Italian utterances.
- 10 intent categories defined in a reference document.
- Stratified sampling ensures balanced representation across all classes.
- Data includes natural language variations, colloquialisms, and diverse phrasing.

**Inference Data:**

- CSV files containing raw user utterances and interaction metadata.
- Pipeline outputs enriched with:
  - Normalized text
  - Predicted intent
  - Confidence scores (0–10 scale)
  - Diagnostic summary (`MATCH`, `MISMATCH`, `UNLISTED`, `NO_MATCH`)

**Note:** Unlabeled utterances were used in an **intent discovery module** to continuously expand the system’s capabilities without exposing sensitive client data.

---

## Methodology

### Intent Classification

1. **Baseline:** Zero-shot classification with pre-trained multilingual NLI models to evaluate task complexity.
2. **Fine-tuned Italian BERT:**
   - **Model:** `dbmdz/bert-base-italian-xxl-cased`
   - **Training strategy:**
     - 5-fold **stratified cross-validation**
     - **Hyperparameter optimization** with Optuna (learning rate, batch size, weight decay)
     - Class weighting for **imbalanced datasets**
   - **Confidence threshold mechanism:** Predictions <0.65 flagged as `UNLISTED` or `NO_MATCH` to prevent low-confidence automation.

### Text Normalization Pipeline

1. **Rule-based preprocessing:** Removes greetings, fillers, and meta-talk phrases using regular expressions.
2. **Generative normalization:** Uses a local **LLaMA-3.2-3B-Instruct model** (4-bit quantized) to clean and formalize utterances based on predicted intents.

---

## Training and Hyperparameter Optimization

- **Dataset split:** 85% train/validation, 15% test.
- **Optimization trials:** 30 Optuna trials to find optimal hyperparameters.
- **Loss function:** Weighted cross-entropy for class imbalance.
- **Metrics monitored:** Accuracy, Weighted F1-Score, confidence score distributions.

---

## Results

| Model                     | Weighted F1 | Accuracy |
|----------------------------|------------|---------|
| Zero-Shot Baseline         | 0.34       | 34%     |
| First BERT Iteration       | 0.78       | 79%     |
| Final Fine-tuned BERT      | 0.89       | 89%     |

**Comparison with Google NLP:**  

- Google NLP Accuracy: 84%  
- BERT Accuracy: 93%  

The fine-tuned Italian BERT demonstrates **superior semantic understanding**, correctly classifying cases where the legacy system failed and effectively filtering ambiguous inputs for discovery.

---

## Usage

This repository includes the **core scripts** required to:

1. Load and fine-tune a BERT model for intent classification.
2. Apply the text normalization pipeline.
3. Generate predictions with confidence scores for evaluation.

**Example:**

```python
from classifier import BertIntentClassifier

classifier = BertIntentClassifier(model_path="models/italian-bert")
prediction = classifier.predict("il terminale del Lotto non mi legge la scheda")
print(prediction)