import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ======================
# PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "results", "final_evaluation.json")

# ======================
# LOAD DATA
# ======================
if not os.path.exists(RESULT_PATH):
    raise FileNotFoundError("❌ Không tìm thấy final_evaluation.json")

with open(RESULT_PATH, "r") as f:
    data = json.load(f)

bert = data["BERT"]
roberta = data["RoBERTa"]

metrics = ["accuracy", "precision", "recall", "f1"]

bert_values = [bert[m] for m in metrics]
roberta_values = [roberta[m] for m in metrics]

# ======================
# PREPARE
# ======================
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

# đóng vòng
bert_values += bert_values[:1]
roberta_values += roberta_values[:1]
angles = np.concatenate([angles, [angles[0]]])

# ======================
# PLOT
# ======================
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# GRID đẹp hơn
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)

# Scale
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

# ======================
# DRAW
# ======================
# BERT
ax.plot(angles, bert_values, linewidth=2, linestyle='solid', label="BERT")
ax.fill(angles, bert_values, alpha=0.15)

# RoBERTa
ax.plot(angles, roberta_values, linewidth=2, linestyle='solid', label="RoBERTa")
ax.fill(angles, roberta_values, alpha=0.25)

# ======================
# VALUE LABEL
# ======================
for i in range(len(metrics)):
    angle = angles[i]

    ax.text(
        angle,
        bert_values[i] + 0.03,
        f"{bert_values[i]:.2f}",
        ha='center',
        fontsize=9
    )

    ax.text(
        angle,
        roberta_values[i] - 0.08,
        f"{roberta_values[i]:.2f}",
        ha='center',
        fontsize=9
    )

# ======================
# TITLE + LEGEND
# ======================
plt.title("BERT vs RoBERTa Performance (Radar Chart)", size=14)

plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

# ======================
# SAVE
# ======================
save_path = os.path.join(BASE_DIR, "results", "radar_chart_pro.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print("✅ Saved to:", save_path)

plt.show()