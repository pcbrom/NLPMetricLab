# NLPMetricLab

**Multilingual NLP Metric Evaluation Toolkit**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://doi.org/10.5281/zenodo.PENDING)

## Overview

NLPMetricLab is a lightweight evaluation toolkit for computing translation quality metrics (BLEU, CHRF, TER, Semantic Similarity) on paired texts. Built as a Streamlit web application, it provides an interactive interface for both batch (CSV/Excel) and manual text comparison, with native support for Chinese (Han character detection + Jieba tokenization).

This tool was developed as the companion evaluation system for the study *Paradox of Poetic Intent in Back-Translation: Evaluating the Quality of Large Language Models in Chinese Translation*, published in Frontiers of Information Technology & Electronic Engineering (FITEE, Springer).

**Online demo:** [nlpmetriclab.streamlit.app](https://nlpmetriclab.streamlit.app/)

## Features

- Upload Excel or CSV files with text data
- Auto-preview and column selection
- Han character detection with Jieba tokenization
- Customizable BLEU weights
- Metric options: BLEU, CHRF, TER, and Semantic Similarity (TF-IDF cosine)
- Download results with concatenated columns and metric outputs
- Manual entry option for comparing individual text pairs

## Installation

```bash
git clone https://github.com/pcbrom/NLPMetricLab.git
cd NLPMetricLab

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
NLPMetricLab/
  app.py               # Main Streamlit application
  metrics.py           # BLEU, CHRF, TER, Semantic Similarity
  tokenizer.py         # Jieba-based CJK preprocessing
  utils.py             # File handling and Han detection
  requirements.txt     # Python dependencies
```

## Citation

If you use this tool, please cite:

```bibtex
@software{weigang2025nlpmetriclab,
  title   = {NLPMetricLab: Multilingual NLP Metric Evaluation Toolkit (v1.0)},
  author  = {Weigang, Li and Brom, Pedro Carvalho},
  year    = {2025},
  url     = {https://github.com/pcbrom/NLPMetricLab},
  license = {CC-BY-NC-SA-4.0},
  note    = {Zenodo DOI pending}
}
```

Related article:

```bibtex
@article{weigang2025paradox,
  title   = {Paradox of Poetic Intent in Back-Translation: Evaluating the
             Quality of Large Language Models in Chinese Translation},
  author  = {Weigang, Li and Brom, Pedro Carvalho},
  journal = {Frontiers of Information Technology \& Electronic Engineering},
  doi     = {10.1631/FITEE.2500298},
  year    = {2025}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0). Commercial use requires a separate written agreement. Contact: pcbrom@gmail.com
