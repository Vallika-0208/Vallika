import os
from rapidfuzz import fuzz
from vectorizer import calculate_similarity

def check_folder_plagiarism(folder_path="."):
    """
    ఇచ్చిన ఫోల్డర్ లోని అన్ని .txt ఫైల్స్ మధ్య ప్లేజియరిజంను చెక్ చేస్తుంది.
    """
    # ఫోల్డర్ లో ఉన్న అన్ని .txt ఫైల్స్ లిస్ట్ తీసుకుంటుంది
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    results = []
    
    # చెక్ చేయడానికి కనీసం 2 ఫైల్స్ ఉండాలి
    if len(files) < 2:
        return results
        
    # ప్రతి ఫైల్‌ను మిగతా అన్ని ఫైల్స్‌తో కంపేర్ చేస్తుంది
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file1_name = files[i]
            file2_name = files[j]
            
            file1_path = os.path.join(folder_path, file1_name)
            file2_path = os.path.join(folder_path, file2_name)
            
            try:
                # ఫైల్స్ ఓపెన్ చేసి టెక్స్ట్ చదువుతుంది
                with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
                    text1 = f1.read()
                    text2 = f2.read()
            except Exception:
                # ఏదైనా ఫైల్ రీడ్ అవ్వకపోతే స్కిప్ చేస్తుంది
                continue
                
            # ఒకవేళ ఫైల్స్ ఖాళీగా ఉంటే స్కిప్ చేస్తుంది
            if not text1.strip() or not text2.strip():
                continue
                
            # Cosine మరియు Fuzzy స్కోర్స్ క్యాలిక్యులేషన్
            cosine_score = calculate_similarity(text1, text2)
            fuzzy_score = fuzz.ratio(text1, text2) / 100.0
            
            # రెండింటి యావరేజ్ స్కోర్ (0.0 నుండి 1.0 మధ్యలో ఉంటుంది)
            final_score = (cosine_score + fuzzy_score) / 2
            
            results.append({
                "file1": file1_name,
                "file2": file2_name,
                "score": final_score
            })
            
    return results