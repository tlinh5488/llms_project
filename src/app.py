import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

BERT_PATH = "results/bert"
ROBERTA_PATH = "results/roberta"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hard-code metrics từ evaluation
BERT_F1 = 0.6730
ROBERTA_F1 = 0.6855

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_models():
    bert_tok = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH).to(DEVICE)

    rob_tok = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    rob_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH).to(DEVICE)

    bert_model.eval()
    rob_model.eval()

    return bert_tok, bert_model, rob_tok, rob_model


bert_tok, bert_model, rob_tok, rob_model = load_models()

# ======================
# PREDICT
# ======================
def predict(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    return probs.cpu().numpy()


# ======================
# UI
# ======================
st.title("📰 Fake News Detection Dashboard")

st.markdown("Compare **BERT vs RoBERTa** for misinformation detection")

text = st.text_area("Enter news title:", height=120)

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("⚠️ Please enter text")
        st.stop()

    bert_probs = predict(text, bert_tok, bert_model)
    rob_probs = predict(text, rob_tok, rob_model)

    bert_pred = "FAKE" if bert_probs[1] > 0.5 else "REAL"
    rob_pred = "FAKE" if rob_probs[1] > 0.5 else "REAL"

    # ======================
    # MODEL COMPARISON
    # ======================
    st.subheader("📊 Model Comparison")

    col1, col2 = st.columns(2)

    # ===== BERT =====
    with col1:
        st.markdown("### 🤖 BERT")

        st.write(f"REAL: {bert_probs[0]:.4f}")
        st.progress(float(bert_probs[0]))

        st.write(f"FAKE: {bert_probs[1]:.4f}")
        st.progress(float(bert_probs[1]))

        if bert_pred == "FAKE":
            st.error(f"Prediction: {bert_pred}")
        else:
            st.success(f"Prediction: {bert_pred}")

    # ===== RoBERTa =====
    with col2:
        st.markdown("### 🚀 RoBERTa")

        st.write(f"REAL: {rob_probs[0]:.4f}")
        st.progress(float(rob_probs[0]))

        st.write(f"FAKE: {rob_probs[1]:.4f}")
        st.progress(float(rob_probs[1]))

        if rob_pred == "FAKE":
            st.error(f"Prediction: {rob_pred}")
        else:
            st.success(f"Prediction: {rob_pred}")

    # ======================
    # CONCLUSION (FIXED LOGIC)
    # ======================
    st.subheader("🧠 Conclusion")

    # Case 1: Agree
    if bert_pred == rob_pred:
        st.success(f"✅ Both models agree: {bert_pred}")

    # Case 2: Disagree
    else:
        st.warning("⚠️ Models disagree → prediction is uncertain")

    # Model performance (global)
    if ROBERTA_F1 > BERT_F1:
        st.info(
            f"📈 RoBERTa performs better overall (F1 = {ROBERTA_F1:.3f} > {BERT_F1:.3f})"
        )
    else:
        st.info(
            f"📈 BERT performs better overall (F1 = {BERT_F1:.3f} > {ROBERTA_F1:.3f})"
        )