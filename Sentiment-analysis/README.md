# AI Restaurant Reviews - Sentiment Analysis NLP Engine

This module implements an isolated, end-to-end Machine Learning and Natural Language Processing (NLP) pipeline developed to parse raw textual customer feedback and accurately classify them into positive or negative categories.

---

##  NLP Pipeline & Machine Learning Architecture

The script processes unstructured text strings through a sequential engineering pipeline to build a robust sentiment classification model:

1. **Text Cleansing (Regex):** Utilizes regular expressions (`re.sub`) to strip out numbers, punctuation, and special characters, leaving only clean alphabetic characters.
2. **Text Normalization:** Converts all characters to lowercase to prevent capitalization disparities during tokenization.
3. **Stopwords Elimination:** Filters out uninformative, high-frequency structural english words (such as "the", "is", "at") using the `nltk.corpus.stopwords` library.
4. **Stemming Engine:** Employs NLTK's `PorterStemmer` to reduce remaining words down to their base morphological root forms (e.g., converting "loved", "loving", and "loves" into the single root "love").
5. **Feature Mapping (TF-IDF Vectorization):** Transforms the cleaned text tokens into structural numerical matrices using a `TfidfVectorizer` to calculate weighted term frequency-inverse document frequencies.
6. **Classification Boundaries:** Trains a Supervised Support Vector Machine model (`LinearSVC`) to isolate and define hyperplanes between positive and negative sentiment classes.

---

##  Local Setup & Dependencies Installation

To run this model locally on your workstation, ensure you have Python 3.11+ configured, and execute the following installation step in your terminal window:

```bash
pip install pandas numpy scikit-learn nltk
