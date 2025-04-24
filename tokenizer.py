import jieba
import pandas as pd

def tokenize_with_jieba(df):
    return df.map(lambda x: " ".join(jieba.lcut(str(x))))
