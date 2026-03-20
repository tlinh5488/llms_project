import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix

# ======================
# PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

BERT_PATH = os.path.join(BASE_DIR, "results", "bert")
ROBERTA_PATH = os.path.join(BASE_DIR, "results", "roberta")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# LOAD TEST SET
# ======================
def load_test_set():
    path = os.path.join(ROBERTA_PATH, "test_set.csv")

    if not os.path.exists(path):
        raise ValueError("❌ Không tìm thấy test_set.csv")

    print("✅ Loaded test set")
    return pd.read_csv(path)


# ======================
# GET PREDICTIONS
# ======================
def get_preds(model, tokenizer, texts):

    model.to(DEVICE)
    model.eval()

    preds = []

    for i in range(0, len(texts), 16):

        batch = texts[i:i+16]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds.extend(pred)

    return preds


# ======================
# PLOT CONFUSION MATRIX (ĐẸP)
# ======================
def plot_cm(cm, title, save_path):

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",  # 🔥 màu đẹp
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        annot_kws={"size": 14}
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    print("✅ Saved:", save_path)

    plt.close()


# ======================
# MAIN
# ======================
def main():

    df = load_test_set()

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    print("\nLoading models...")

    # ======================
    # BERT
    # ======================
    print("Evaluating BERT...")

    bert_tok = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)

    bert_preds = get_preds(bert_model, bert_tok, texts)
    cm_bert = confusion_matrix(labels, bert_preds)

    plot_cm(
        cm_bert,
        "BERT Confusion Matrix",
        os.path.join(BASE_DIR, "results", "bert_cm.png")
    )

    # ======================
    # RoBERTa
    # ======================
    print("Evaluating RoBERTa...")

    rob_tok = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    rob_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH)

    rob_preds = get_preds(rob_model, rob_tok, texts)
    cm_rob = confusion_matrix(labels, rob_preds)

    plot_cm(
        cm_rob,
        "RoBERTa Confusion Matrix",
        os.path.join(BASE_DIR, "results", "roberta_cm.png")
    )


if __name__ == "__main__":
    main()