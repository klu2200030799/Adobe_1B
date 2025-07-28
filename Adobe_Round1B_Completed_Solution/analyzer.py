import os
import json
import fitz
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if 'lines' in b:
                text = " ".join(s['text'] for l in b['lines'] for s in l['spans']).strip()
                if len(text.split()) > 5:
                    sections.append({
                        "doc": os.path.basename(pdf_path),
                        "page": page_num,
                        "text": text
                    })
    return sections

def rank_sections(sections, query):
    corpus = [s['text'] for s in sections] + [query]
    tfidf = TfidfVectorizer().fit_transform(corpus)
    scores = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    ranked = sorted(zip(sections, scores), key=lambda x: -x[1])
    top_ranked = []
    for i, (sec, score) in enumerate(ranked[:10]):
        sec['importance_rank'] = i + 1
        sec['score'] = float(score)
        top_ranked.append(sec)
    return top_ranked

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    input_dir = "/app/input"
    persona = load_json(os.path.join(input_dir, "persona.json"))
    job = load_json(os.path.join(input_dir, "job.json"))
    query = persona['role'] + ". " + persona['focus'] + ". " + job['task']

    all_sections = []
    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            all_sections.extend(extract_sections(pdf_path))

    ranked_sections = rank_sections(all_sections, query)

    output = {
        "metadata": {
            "documents": [f for f in os.listdir(input_dir) if f.endswith(".pdf")],
            "persona": persona,
            "job": job,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "sections": ranked_sections
    }

    with open("/app/output/output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
