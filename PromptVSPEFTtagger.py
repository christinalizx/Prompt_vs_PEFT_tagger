import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import os
from tqdm import tqdm

# To disable the huggingface/tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Stage 1: Data Preparation ---
class DataPreparer:
    """
    Handles loading, cleaning, and splitting the dataset.
    """
    def __init__(self, file_path, random_seed=42):
        self.file_path = file_path
        self.random_seed = random_seed

    def load_and_prepare(self):
        """Loads and prepares the dataset from the CSV file."""
        print("--- Starting Stage 1: Data Preparation ---")
        self.df = pd.read_csv(f"{self.file_path}.csv")
        self.df = self.df.rename(columns={"code": "utterance", "theme": "label"})
        self.df["utterance"] = self.df["utterance"].astype(str).apply(lambda x: re.sub(r"^\[.*?\]\s*", "", x).strip())
        self.df.dropna(subset=['label', 'utterance'], inplace=True)

        theme_map = {
            'communication through tech': 'Communicating through technology',
            'issues in communication due to technology': 'Communicating through technology',
        }
        self.df['label'] = self.df['label'].replace(theme_map)
        print("Theme optimization complete with new mapping.")

        self.df["label"] = pd.Categorical(self.df["label"])
        self.themes = list(self.df['label'].cat.categories)
        self.df["label"] = self.df["label"].cat.codes

        train_val_df, test_df = train_test_split(
            self.df, test_size=0.15, random_state=self.random_seed, stratify=self.df["label"]
        )
        train_df, dev_df = train_test_split(
            train_val_df, test_size=0.15, random_state=self.random_seed, stratify=train_val_df["label"]
        )

        self.dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(dev_df),
            "test": Dataset.from_pandas(test_df),
        })

        class_label_feature = ClassLabel(names=self.themes)
        self.dataset = self.dataset.cast_column("label", class_label_feature)

        print("Data preparation complete.")
        print(f"Final themes after re-mapping: {self.themes}")
        return self.dataset, self.themes

# --- Stage 2: Prompt-Only Baseline ---
class PromptOnlyBaseline:
    """
    Evaluates zero-shot, four-shot, and chain-of-thought prompting baselines.
    """
    def __init__(self, model_name, themes):
        self.model_name = model_name
        self.themes = themes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_prediction(self, prompt):
        """Generates a response and parses the theme."""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        for i, theme in enumerate(self.themes):
            if theme.lower() in generated_text.lower():
                return i
        return -1

    def evaluate(self, test_dataset, strategy="zero_shot"):
        print(f"\n--- Evaluating Prompt-Only Baseline ({strategy}) ---")
        predictions, true_labels = [], []
        
        four_shot_examples = None
        if strategy == "four_shot":
            four_shot_examples = self.dataset['train'].shuffle(seed=42).select(range(4))

        for item in tqdm(test_dataset, desc=f"Running {strategy} evaluation"):
            utterance = item['utterance']
            true_label = item['label']

            if strategy == "zero_shot":
                prompt = f"Classify the following utterance into one of these themes: {', '.join(self.themes)}. Utterance: '{utterance}'\nTheme:"
            elif strategy == "four_shot":
                 example_str = "\n".join([f"Utterance: '{ex['utterance']}'\nTheme: {self.themes[ex['label']]}" for ex in four_shot_examples])
                 prompt = f"{example_str}\n\nUtterance: '{utterance}'\nTheme:"
            elif strategy == "chain_of_thought":
                prompt = f"Let's think step by step. The themes are: {', '.join(self.themes)}. The utterance is: '{utterance}'. Analyze the utterance and determine the most fitting theme.\nReasoning: The utterance is about... Therefore, the best theme is "

            predicted_label = self._get_prediction(prompt)
            predictions.append(predicted_label)
            true_labels.append(true_label)
            
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)

        print(f"Results for {strategy}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro-F1 Score: {f1:.4f}")
        return {"accuracy": accuracy, "f1_macro": f1}

# --- Stage 3: Fine-Tuning (PEFT) ---
def compute_metrics(eval_pred):
    """Computes accuracy and F1 score for the Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": accuracy, "f1_macro": f1}

class LoRATrainer:
    """
    Handles the fine-tuning of the model using LoRA.
    """
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, tokenized_dataset):
        """Trains the model."""
        print("\n--- Starting Stage 3: PEFT (LoRA) ---")
        num_labels = tokenized_dataset["train"].features["label"].num_classes

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        model.config.pad_token_id = model.config.eos_token_id
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir="./results-lora",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            use_mps_device=torch.backends.mps.is_available(),
            report_to="none",
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        print("PEFT training complete.")
        return trainer

# --- Main Execution ---
if __name__ == "__main__":
    FILE_PATH = "codes-v1(sorted)"
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # 1. Data Preparation
    data_preparer = DataPreparer(FILE_PATH)
    dataset, themes_from_data = data_preparer.load_and_prepare()

    if dataset:
        # 2. Prompt-Only Baselines
        prompt_baseline_evaluator = PromptOnlyBaseline(MODEL_NAME, themes_from_data)
        prompt_baseline_evaluator.dataset = dataset # Pass dataset for four-shot sampling
        prompt_baseline_evaluator.evaluate(dataset['test'], strategy="zero_shot")
        prompt_baseline_evaluator.evaluate(dataset['test'], strategy="four_shot")
        prompt_baseline_evaluator.evaluate(dataset['test'], strategy="chain_of_thought")

        # 3. Tokenize the dataset for the main model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        def tokenize_function(examples):
            return tokenizer(examples["utterance"], truncation=True, max_length=128)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["utterance"])

        # 4. Fine-Tune LoRA model
        lora_trainer_instance = LoRATrainer(model_name=MODEL_NAME)
        trained_lora_model = lora_trainer_instance.train(tokenized_dataset)

        # 5. Print and Save Final LoRA Model Results
        print("\n--- Evaluating Final LoRA Model on Test Set ---")
        final_results = trained_lora_model.evaluate(tokenized_dataset["test"])
        print("\n--- Final LoRA Model Performance ---")
        for key, value in final_results.items():
            if "eval" in key:
                print(f"{key.replace('eval_', '').capitalize()}: {value:.4f}")
        with open("final_lora_evaluation.json", "w") as f:
            json.dump(final_results, f, indent=4)
        print("\nFull evaluation results saved to final_lora_evaluation.json")

        # 6. Print LoRA Performance Per Epoch
        print("\n--- LoRA Performance Per Epoch (on Validation Set) ---")
        for log in trained_lora_model.state.log_history:
            if 'eval_loss' in log: # Filter for evaluation logs
                epoch = log.get('epoch', 'N/A')
                accuracy = log.get('eval_accuracy', 'N/A')
                f1_macro = log.get('eval_f1_macro', 'N/A')
                print(f"Epoch {epoch:.2f}: Accuracy = {accuracy:.4f}, Macro-F1 = {f1_macro:.4f}")