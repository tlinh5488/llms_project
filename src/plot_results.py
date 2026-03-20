import json
import os
import matplotlib.pyplot as plt

# ======================
# PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "results", "final_evaluation.json")

# ======================
# LOAD JSON
# ======================
if not os.path.exists(RESULT_PATH):
    raise FileNotFoundError("❌ Không tìm thấy file final_evaluation.json")

with open(RESULT_PATH, "r") as f:
    data = json.load(f)

bert = data["BERT"]
roberta = data["RoBERTa"]

metrics = ["accuracy", "precision", "recall", "f1"]

bert_values = [bert[m] for m in metrics]
roberta_values = [roberta[m] for m in metrics]

# ======================
# PLOT
# ======================
x = range(len(metrics))

plt.figure(figsize=(8, 5))

# BERT
bars1 = plt.bar(
    [i - 0.2 for i in x],
    bert_values,
    width=0.4,
    label="BERT"
)

# RoBERTa
bars2 = plt.bar(
    [i + 0.2 for i in x],
    roberta_values,
    width=0.4,
    label="RoBERTa"
)

# Hiển thị giá trị trên cột
for bar in bars1:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.01, f"{y:.2f}", ha='center')

for bar in bars2:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.01, f"{y:.2f}", ha='center')

# Labels
plt.xticks(x, [m.upper() for m in metrics])
plt.ylim(0, 1)

plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("BERT vs RoBERTa Performance Comparison")

plt.legend()

plt.tight_layout()

# ======================
# SAVE
# ======================
save_path = os.path.join(BASE_DIR, "results", "comparison.png")
plt.savefig(save_path, dpi=300)

print("✅ Saved chart to:", save_path)

plt.show()