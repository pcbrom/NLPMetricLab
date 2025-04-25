import streamlit as st
import pandas as pd
from utils import detect_han_characters, load_file
from tokenizer import tokenize_with_jieba
from metrics import compute_metrics

st.set_page_config(page_title="NLPMetricLab", layout="wide")
st.title("📊 NLP Metric Lab")
st.markdown("[Powered by PCBrom & LiWeigang](https://github.com/pcbrom/NLPMetricLab)")

# Tabs
tab1, tab2 = st.tabs(["Data Input", "Methodology"])

with tab1:
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
            data = pd.DataFrame([[col1_input, col2_input]], columns=["original_text", "another_text"])
            result_df, results_summary = compute_metrics(data, "original_text", "another_text", metrics_selected, weights)
            st.subheader("Results")
            st.dataframe(result_df)

with tab2:
    st.header("Methodology")
    st.subheader("Bilingual Evaluation Understudy (BLEU)")
    st.markdown("""
        BLEU is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. 
        Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is". 
        BLEU was one of the first metrics to claim a high correlation with human judgements of quality
    """)

    st.subheader("Character n-gram F-score (CHRF)")
    st.markdown("""
        CHRF is a metric for evaluating the quality of machine-generated text. It calculates a character n-gram F-score between the generated text and a reference text.
    """)

    st.subheader("Translation Edit Rate (TER)")
    st.markdown("""
        TER is an error metric for machine translation that measures the number of edits required to change a system output into one of the references.
    """)

    st.subheader("Semantic Similarity")
    st.markdown("""
        Semantic similarity measures the degree to which two texts carry the same meaning. This implementation uses TF-IDF vectors and cosine similarity.
    """)
