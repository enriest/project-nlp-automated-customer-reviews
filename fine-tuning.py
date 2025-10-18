from sklearn.utils.class_weight import compute_class_weight
# -------------------------------------------------------------
# USAGE EXAMPLE: How to use this module from your notebook
# -------------------------------------------------------------
#
# from fine_tuning import run_fine_tuning, get_device
# device = get_device()
# fine_tuned_results = run_fine_tuning(transformer_data, models_to_finetune, device)
#
# Arguments:
#   transformer_data: dict of preprocessed data for each model
#   models_to_finetune: dict of model names to HuggingFace IDs or local paths
#   device: torch.device (optional, will auto-detect if not provided)
#
# Returns:
#   fine_tuned_results: dict of results for each model
#
# You can then use fine_tuned_results for evaluation and comparison in your notebook.

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fine_tune_model(model_name, model_id, data_dict, device):
    print(f"\nFINE-TUNING {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.to(device)
    train_dataset = SentimentDataset(data_dict['train_encodings'], data_dict['y_train'])
    test_dataset = SentimentDataset(data_dict['test_encodings'], data_dict['y_test'])
    # Compute class weights for this train split
    y_np = data_dict['y_train'].cpu().numpy()
    weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1,2]), y=y_np)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    # Custom Trainer with class-weighted loss
    from transformers import Trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss
    training_args = TrainingArguments(
        output_dir=f"./offline_models/{model_name.lower()}_finetuned",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
        load_best_model_at_end=False
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'training_loss': train_result.training_loss,
        'eval_loss': eval_result['eval_loss'],
        'y_pred': pred_labels,
        'y_true': predictions.label_ids
    }

def run_fine_tuning(transformer_data, models_to_finetune, device=None):
    if device is None:
        device = get_device()
    results = {}
    for model_name, model_id in models_to_finetune.items():
        data_dict = transformer_data.get(model_name)
        if data_dict is None:
            print(f"Skipping {model_name}: no data available.")
            continue
        result = fine_tune_model(model_name, model_id, data_dict, device)
        results[model_name] = result
    return results
