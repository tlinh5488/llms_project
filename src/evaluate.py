import os
import json
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ======================
# PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results")

BERT_PATH = os.path.join(RESULTS_DIR, "bert")
ROBERTA_PATH = os.path.join(RESULTS_DIR, "roberta")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# METRICS
# ======================
def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# ======================
# LOAD TEST SET
# ======================
def load_test_set():

    test_path = os.path.join(ROBERTA_PATH, "test_set.csv")

    if not os.path.exists(test_path):
        raise ValueError("❌ Không tìm thấy test_set.csv (train lại RoBERTa)")

    print("✅ Using saved test set")

    return pd.read_csv(test_path)


# ======================
# EVALUATE MODEL
# ======================
def evaluate_model(model, tokenizer, test_df, batch_size=16):

    model.to(DEVICE)
    model.eval()

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    preds = []

    for i in range(0, len(texts), batch_size):

        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

        preds.extend(batch_preds)

    return compute_metrics(labels, preds)


# ======================
# MAIN
# ======================
def main():

    print("Loading test set...")
    test_df = load_test_set()
    print("Test samples:", len(test_df))

    print("\nLoading models...")

    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)

    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH)

    # ======================
    # EVALUATE
    # ======================
    print("\nEvaluating BERT...")
    bert_results = evaluate_model(bert_model, bert_tokenizer, test_df)
    print("BERT:", bert_results)

    print("\nEvaluating RoBERTa...")
    roberta_results = evaluate_model(roberta_model, roberta_tokenizer, test_df)
    print("RoBERTa:", roberta_results)

    # ======================
    # SAVE JSON
    # ======================
    final_results = {
        "BERT": bert_results,
        "RoBERTa": roberta_results
    }

    save_path = os.path.join(RESULTS_DIR, "final_evaluation.json")

    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n✅ Saved to:", save_path)


if __name__ == "__main__":
    main()