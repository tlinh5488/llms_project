================📰 Fake News Detection with BERT vs RoBERTa=================

A Natural Language Processing project that compares BERT and RoBERTa for fake news classification using real-world datasets.


🔍 Confusion Matrix

BERT

<img width="927" height="775" alt="image" src="https://github.com/user-attachments/assets/71c83610-c9ff-4899-ac5e-4890fabdef79" />


RoBERTa

<img width="925" height="769" alt="image" src="https://github.com/user-attachments/assets/2beb10f7-57a6-47be-ad64-f42bb51d6b52" />


🚀 Features

Train and compare BERT vs RoBERTa

Evaluate with standard metrics:

Accuracy

Precision

Recall

F1-score

Visualization:

Bar chart

Radar chart

Confusion matrix

Interactive Streamlit app

Reproducible pipeline

📊 Results

<img width="813" height="257" alt="image" src="https://github.com/user-attachments/assets/88314340-e6ab-4095-b652-012bde38d629" />

<img width="1042" height="655" alt="image" src="https://github.com/user-attachments/assets/d0cde7b2-fa59-4ab9-8f63-66d912fe663c" />

📂 Dataset

This project uses data derived from:

FakeNewsNet
🔗 https://github.com/KaiDMML/FakeNewsNet

Included subsets:

GossipCop

Politifact

📌 Data fields used:

title → input text

label → 0 (REAL), 1 (FAKE)

⚙️ Installation
pip install torch transformers datasets scikit-learn pandas matplotlib streamlit accelerate

🧠 Usage
1. Build dataset

python src/build_dataset.py

2. Train models

python src/train_bert.py

python src/train_roberta.py

3. Evaluate

python src/evaluate.py

4. Visualization

python src/plot_results.py

python src/confusion_matrix.py

5. Run app

streamlit run app.py

🖥 Demo

Example input:

"Breaking: Scientists confirm aliens landed in New York"

Output:

BERT → FAKE

RoBERTa → FAKE (higher confidence)

📌 Key Insights

BERT

Higher precision

Better at avoiding false positives

RoBERTa

Higher recall

Better at detecting fake news

👉 Overall: RoBERTa generalizes better

🔮 Future Work

Use full dataset from FakeNewsNet

Add LLM models (LLaMA, GPT)

Multi-modal learning (text + image)

Hyperparameter tuning

Data augmentation

🛠 Tech Stack

PyTorch

HuggingFace Transformers

Scikit-learn

Streamlit

Matplotlib

👨‍💻 Author

NLP Fake News Detection Project

Contact: [tlinh5488](https://github.com/tlinh5488) ( Thuy Linh )

⭐ Notes

If this project helps you, consider giving it a ⭐ on GitHub.
