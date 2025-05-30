import streamlit as st
import pandas as pd
from utils import detect_han_characters, load_file
from tokenizer import tokenize_with_jieba
from metrics import compute_metrics

st.set_page_config(page_title="NLPMetricLab", layout="wide")
st.title("📊 NLP Metric Lab")
st.markdown("[Powered by PCBrom & LiWeigang](https://github.com/pcbrom/NLPMetricLab)")

# Tabs
tab1, tab2, tab3 = st.tabs(["Data Input", "Methodology", "Arxiv"])

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

with tab3:
    st.header("The Paradox of Poetic Intent in Back-Translation: Evaluating the Quality of Large Language Models in Chinese Translation")
    st.markdown("[https://arxiv.org/abs/2504.16286](https://arxiv.org/abs/2504.16286)")
    st.markdown("Li Weigang, Pedro Carvalho Brom")
    st.write("""
        The rapid advancement of large language models (LLMs) has reshaped the landscape of machine translation, yet challenges persist in preserving poetic intent, cultural heritage, and handling specialized terminology in Chinese-English translation. 
        This study constructs a diverse corpus encompassing Chinese scientific terminology, historical translation paradoxes, and literary metaphors. 
        Utilizing a back-translation and Friedman test-based evaluation system (BT-Fried), we evaluate BLEU, CHRF, TER, and semantic similarity metrics across six major LLMs (e.g., GPT-4.5, DeepSeek V3) and three traditional translation tools. 
        Key findings include: (1) Scientific abstracts often benefit from back-translation, while traditional tools outperform LLMs in linguistically distinct texts; (2) LLMs struggle with cultural and literary retention, exemplifying the "paradox of poetic intent"; (3) Some models exhibit "verbatim back-translation", reflecting emergent memory behavior; (4) A novel BLEU variant using Jieba segmentation and n-gram weighting is proposed. 
        The study contributes to the empirical evaluation of Chinese NLP performance and advances understanding of cultural fidelity in AI-mediated translation.
    """)
