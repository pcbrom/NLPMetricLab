import streamlit as st
import pandas as pd
from utils import detect_han_characters, load_file
from tokenizer import tokenize_with_jieba
from metrics import compute_metrics

st.set_page_config(page_title="NLPMetricLab", layout="wide")
st.title("📊 NLP Metric Lab")
st.markdown("[Powered by PCBrom & LiWeigang](https://github.com/pcbrom/NLPMetricLab)")

option = st.radio("Choose input method:", ["📂 Upload Excel/CSV", "⌨️ Manual Input"])

if option == "📂 Upload Excel/CSV":
    uploaded_file = st.file_uploader("Upload a file (Excel or CSV)", type=["xlsx", "csv"])
    if uploaded_file:
        df = load_file(uploaded_file)
        st.dataframe(df.head())

        columns = st.multiselect("Select **two** text columns to compare", df.columns, max_selections=2)
        if len(columns) == 2:
            col1, col2 = columns

            if detect_han_characters(df[[col1, col2]]):
                st.warning("Han characters detected. Applying Jieba tokenizer.")
                df[[col1, col2]] = tokenize_with_jieba(df[[col1, col2]])

            metrics_selected = st.multiselect("Select metrics", ["BLEU", "CHRF", "TER", "Semantic Sim"])
            bleu_weights_input = st.text_input("BLEU Weights (n-gram):", "0.25, 0.25, 0.25, 0.25")
            try:
                weights = tuple(map(float, bleu_weights_input.split(",")))
            except:
                st.error("Invalid weights")
                weights = (0.25, 0.25, 0.25, 0.25)

            if st.button("Compute Metrics"):
                df_subset = df[[col1, col2]].copy()
                result_df, results_summary = compute_metrics(df_subset, col1, col2, metrics_selected, weights)
                st.subheader("Results")
                st.dataframe(result_df)
                st.download_button("📥 Download Results", result_df.to_csv(index=False).encode("utf-8"), file_name="results.csv")

else:
    st.subheader("Manual Text Entry")
    col1_input = st.text_area("Original Text")
    col2_input = st.text_area("Translated Text")

    metrics_selected = st.multiselect("Select metrics", ["BLEU", "CHRF", "TER", "Semantic Sim"])
    bleu_weights_input = st.text_input("BLEU Weights (n-gram):", "0.25, 0.25, 0.25, 0.25")
    try:
        weights = tuple(map(float, bleu_weights_input.split(",")))
    except:
        st.error("Invalid weights")
        weights = (0.25, 0.25, 0.25, 0.25)

    if st.button("Compute Metrics"):
        data = pd.DataFrame([[col1_input, col2_input]], columns=["original_text", "en_zh"])
        result_df, results_summary = compute_metrics(data, "original_text", "en_zh", metrics_selected, weights)
        st.subheader("Results")
        st.dataframe(result_df)
