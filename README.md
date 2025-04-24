# NLPMetricLab

NLPMetricLab is a lightweight Streamlit application designed for evaluating similarity between pairs of text using multiple metrics, including BLEU, CHRF, TER and Semantic Similarity. It supports multilingual inputs and includes automatic handling of Han characters via Jieba tokenizer.

**Authors:** PCBrom & LiWeigang

## 🚀 Features

* 📂 Upload Excel or CSV files with text data.
* 📊 Auto-preview and column selection.
* 🈶 Han character detection with Jieba tokenization.
* ✅ Customizable BLEU weights.
* 📐 Metric options: BLEU, CHRF, TER, and Semantic Similarity (TF-IDF based).
* 📥 Download results with concatenated columns and metric outputs.
* ⌨️ Manual entry option for comparing individual text pairs.

## 📁 Project Structure

```
TextMetricLab/
├── app.py               # Main Streamlit app
├── utils.py             # File handling and Han detection
├── tokenizer.py         # Jieba-based preprocessing
├── metrics.py           # Metric calculation functions
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
```

## ⚙️ Installation

```bash
# Clone the repository
https://github.com/yourusername/NLPMetricLab.git
cd TextMetricLab

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📌 Notes

* Ensure your CSV or Excel file contains at least two textual columns.
* BLEU weights can be adjusted manually via the input field.
* Semantic Similarity is computed using TF-IDF vectors and cosine similarity.
* You can also input text manually without uploading a file.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
