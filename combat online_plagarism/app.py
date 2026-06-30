# app.py లో పాతది అంతా తీసేసి ఈ ఒక్క కోడ్ మాత్రమే పేస్ట్ చెయ్
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# 1. Similarity క్యాలిక్యులేట్ చేసే ఫంక్షన్
def calculate_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

# 2. ఫోల్డర్ స్కాన్ చేసే ఫంక్షన్
def check_folder_plagiarism(folder_path="."):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    results = []
    
    if len(files) < 2:
        return results
        
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file1_name = files[i]
            file2_name = files[j]
            
            file1_path = os.path.join(folder_path, file1_name)
            file2_path = os.path.join(folder_path, file2_name)
            
            try:
                with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
                    text1 = f1.read()
                    text2 = f2.read()
            except Exception:
                continue
                
            if not text1.strip() or not text2.strip():
                continue
                
            cosine_score = calculate_similarity(text1, text2)
            fuzzy_score = fuzz.ratio(text1, text2) / 100.0
            final_score = (cosine_score + fuzzy_score) / 2
            
            results.append({
                "file1": file1_name,
                "file2": file2_name,
                "score": final_score
            })
            
    return results

# 3. Streamlit వెబ్‌పేజీ UI కాన్ఫిగレーション
st.set_page_config(page_title="AI Folder Plagiarism Checker", page_icon="📂", layout="wide")

st.title("📂 AI Folder Plagiarism Checker")
st.write("ఈ యాప్ మీ ప్రాజెక్ట్ ఫోల్డర్‌లో ఉన్న అన్ని `.txt` ఫైల్స్ మధ్య ప్లేజియరిజంను ఒకేసారి చెక్ చేస్తుంది.")
st.markdown("---")

if st.button("🚀 Scan Folder For Plagiarism", use_container_width=True):
    with st.spinner("ఫోల్డర్‌లోని అన్ని ఫైల్స్‌ను స్క్యాన్ చేస్తున్నాము... దయచేసి వేచి ఉండండి..."):
        results = check_folder_plagiarism(".") 
        
        if not results:
            st.info("💡 చెక్ చేయడానికి ఫోల్డర్‌లో కనీసం రెండు `.txt` ఫైల్స్ ఉండాలి! (file1.txt, file2.txt లాగా క్రియేట్ చేయండి)")
        else:
            st.subheader("📊 స్కానింగ్ రిపోర్ట్ (Scan Results)")
            
            table_data = []
            for res in results:
                percentage = f"{res['score'] * 100:.2f}%"
                status = "⚠️ Plagiarism Detected" if res['score'] > 0.5 else "✅ Unique"
                
                table_data.append({
                    "File 1": res['file1'],
                    "File 2": res['file2'],
                    "Similarity Score": percentage,
                    "Status": status
                })
            
            st.table(table_data)
