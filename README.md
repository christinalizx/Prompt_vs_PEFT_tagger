# Thematic Analysis of Interview Data: Prompting vs. Fine-Tuning

This project investigates and compares two lightweight computational methods for automating the thematic coding of qualitative interview data: in-context learning via prompting and parameter-efficient fine-tuning (PEFT) with LoRA.

The goal is to provide a direct comparison between these resource-friendly approaches to determine if careful prompting of a base language model is sufficient for nuanced classification, or if a small amount of fine-tuning offers a significant performance advantage.

---

## 📋 Project Overview

This repository contains a Python script that implements a full pipeline for this comparison:

1.  **Data Preparation**: Cleans and preprocesses raw interview data from a CSV file, handling label mapping and splitting the data for training, validation, and testing.
2.  **Prompt-Only Baselines**: Evaluates the performance of a base language model (`TinyLlama-1.1B-Chat-v1.0`) on the classification task using zero-shot, four-shot, and chain-of-thought prompting strategies.
3.  **LoRA Fine-Tuning**: Fine-tunes the same base model for sequence classification using Low-Rank Adaptation (LoRA), a parameter-efficient technique.
4.  **Evaluation**: Compares the final accuracy and macro F1-score of the fine-tuned model against the prompt-only baselines.
5.  **Analysis**: Generates reports and visualizations, including a loss curve graph, to analyze model performance.

---

## 📊 Dataset

The project uses the `codes-v1(sorted).csv` dataset, which contains sentences (`utterance`) from interviews with aging parents and their adult children, each manually assigned a `theme`. The dataset exhibits a significant class imbalance, which is a key challenge addressed in the project.

---

## 📈 Results Summary
The experiments provide a clear conclusion: fine-tuning is substantially more effective than prompting for this task.

Prompt-Only Baselines: The best-performing prompt-only method (zero-shot) achieved an accuracy of only 17.2%. The models struggled to understand the nuanced themes from the prompt context alone.

Fine-Tuned LoRA Model: The LoRA model achieved a final accuracy of 53.1% and a macro F1-score of 26.4%. This represents a 3x improvement in accuracy over the best baseline.

While the fine-tuned model's performance is still constrained by the dataset's limitations, the results validate that adapting the model's weights via PEFT is a superior strategy for specialized classification tasks.

## ⚠️ Limitations
Data Imbalance: The dataset has a severe class imbalance, which is the primary reason for the low macro F1-score. The model performs well on the majority class but struggles with under-represented themes.

Small Dataset Size: The limited number of examples increases the risk of overfitting and makes it difficult for the model to learn generalizable patterns for all themes.

Task Complexity: Classifying nuanced human communication is an inherently difficult task, and some sentences may contain overlapping themes.

## 🔮 Future Work
Based on the project findings, future work could explore:

Data Augmentation: Using techniques like oversampling (e.g., SMOTE) or back-translation to address the class imbalance.

Advanced Prompting: Designing more sophisticated prompts for the baseline models, perhaps including detailed definitions for each theme.

Hyperparameter Tuning: Experimenting with different LoRA configurations (e.g., rank, alpha) or training parameters (e.g., learning rate) to further optimize the fine-tuned model.
