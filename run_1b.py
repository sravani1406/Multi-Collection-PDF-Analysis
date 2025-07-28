import os
import fitz  # PyMuPDF
import json
import time
from sentence_transformers import SentenceTransformer, util

# Load from local offline model directory
model = SentenceTransformer("./model")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    content = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            content.append({"page": i + 1, "text": text})
    return content

def rank_sections_by_similarity(content, persona_job_text):
    results = []
    persona_emb = model.encode(persona_job_text)

    for section in content:
        section_text = section["text"]
        section_emb = model.encode(section_text)
        similarity = util.cos_sim(persona_emb, section_emb).item()
        results.append({
            "document": section["doc"],
            "page": section["page"],
            "section_title": f"Page {section['page']}",
            "importance_rank": similarity,
            "refined_text": section_text
        })
    
    results.sort(key=lambda x: x["importance_rank"], reverse=True)
    return results[:5]

def process_documents(input_dir, persona, job):
    persona_job_text = persona + ". " + job
    extracted_content = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(input_dir, filename)
            pages = extract_text_from_pdf(filepath)
            for p in pages:
                p["doc"] = filename
                extracted_content.append(p)

    ranked_sections = rank_sections_by_similarity(extracted_content, persona_job_text)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "metadata": {
            "documents": [f for f in os.listdir(input_dir) if f.endswith(".pdf")],
            "persona": persona,
            "job_to_be_done": job,
            "processed_at": timestamp
        },
        "extracted_sections": ranked_sections
    }

def run(input_dir="input", output_dir="output", persona_file="persona.json"):
    os.makedirs(output_dir, exist_ok=True)
    with open(persona_file, "r", encoding="utf-8") as f:
        persona_data = json.load(f)

    result = process_documents(
        input_dir,
        persona=persona_data["persona"],
        job=persona_data["job_to_be_done"]
    )

    with open(os.path.join(output_dir, "challenge1b_output.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    run()
