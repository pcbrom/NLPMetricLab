import pandas as pd
import jieba
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Tokenization function for Chinese texts using Jieba
def tokenize_chinese_jieba(text: str) -> list:
    return list(jieba.cut(str(text)))

# BLEU (custom weights)
def calculate_bleu(candidate_tokens, reference_tokens, weights=(0.25, 0.25, 0.25, 0.25)):
    try:
        return sentence_bleu([reference_tokens], candidate_tokens, weights=weights)
    except Exception:
        return 0.0

# CHRF (character n-gram F-score, simplificado)
def calculate_chrf(candidate_tokens, reference_tokens):
    candidate_str = "".join(candidate_tokens)
    reference_str = "".join(reference_tokens)
    return len(set(candidate_str) & set(reference_str)) / len(set(reference_str)) if len(set(reference_str)) > 0 else 0.0

# TER (approx via TF-IDF + MSE)
def calculate_ter(candidate_text: str, reference_text: str):
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([candidate_text, reference_text])
        return mean_squared_error(tfidf_matrix[0].toarray(), tfidf_matrix[1].toarray())
    except Exception:
        return 0.0

# Semantic Similarity via TF-IDF + cosine
def calculate_semantic_similarity(original: str, translated: str):
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([original, translated])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except Exception:
        return 0.0

# Apply all metrics row-wise
def calculate_metrics_for_row(row, weights=(0.25, 0.25, 0.25, 0.25)):
    original_text = row['original_text']
    translated_text = row['another_text']

    original_tokens = tokenize_chinese_jieba(original_text)
    translated_tokens = tokenize_chinese_jieba(translated_text)

    bleu_value = calculate_bleu(translated_tokens, original_tokens, weights=weights)
    chrf_value = calculate_chrf(translated_tokens, original_tokens)
    ter_value = calculate_ter("".join(translated_tokens), "".join(original_tokens))
    semantic_similarity = calculate_semantic_similarity(original_text, translated_text)

    return pd.Series([bleu_value, chrf_value, ter_value, semantic_similarity], 
                     index=["BLEU", "CHRF", "TER", "SemanticSim"])

# DataFrame-wide metric computation
def compute_metrics(df: pd.DataFrame, col1: str, col2: str, metrics_selected: list[str], weights=(0.25, 0.25, 0.25, 0.25)) -> tuple[pd.DataFrame, dict]:
    df = df.rename(columns={col1: 'original_text', col2: 'another_text'})  # adapt to internal names
    df_metrics = df.apply(lambda row: calculate_metrics_for_row(row, weights=weights), axis=1)

    # Merge results back
    df = pd.concat([df, df_metrics], axis=1)

    # Aggregate for summary
    results = {metric: df_metrics[metric].mean() for metric in df_metrics.columns if metric in metrics_selected}
    return df, results
