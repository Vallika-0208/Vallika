# 📂 AI-Powered Folder Plagiarism Checker

An interactive web application built with **Streamlit** that scans and detects plagiarism across all `.txt` files within a project folder simultaneously. It uses a hybrid approach combining **TF-IDF + Cosine Similarity** and **Fuzzy String Matching** for highly accurate results.

---

## 🚀 Features
- **Folder-wide Scanning:** Compares every `.txt` file with every other file in the directory.
- **Hybrid Detection:** Uses Machine Learning (TF-IDF) for vocabulary overlap and Fuzzy Matching for structural similarity.
- **User-Friendly UI:** Clean, interactive table display with plagiarism alerts using Streamlit.
- **Telugu Language Support:** The application interface text is designed in Telugu for localized usability.

---

## 🛠️ Tech Stack & Algorithms
- **Frontend:** Streamlit
- **Text Vectorization:** Scikit-learn (TfidfVectorizer)
- **Similarity Metric:** Cosine Similarity
- **String Matching:** RapidFuzz (Fuzzy Ratio)

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/AI_Plagiarism_Checker.git](https://github.com/YOUR_GITHUB_USERNAME/AI_Plagiarism_Checker.git)
   cd AI_Plagiarism_Checker
