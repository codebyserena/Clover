import pdfplumber
from docx import Document
from typing import Dict

def extract_text_from_pdf(file_path:str)->str:
    text=""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text=page.extract_text()
                if page_text:
                    text+=page_text+"\n"
    except Exception as e:
        print("Error extracting text from pdf: {e}")
    return text.strip()

def extract_text_from_docx(file_path:str)->str:
    text=""
    try:
        doc=Document(file_path)
        for para in doc.paragraphs:
            text+=para.text+"\n"
    except Exception as e:
        print("Error extracting text from docx:{e}")
    return text.strip()

def extract_text(file_path:str)->str:
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are allowed.")

def extract_entities(text: str) -> Dict:
    return {
        "skills": [],
        "job_titles": [],
        "companies": [],
        "dates": []
    }

def build_user_profile_json(entities: Dict, preferences_text: str) -> Dict:
    return {
        "id": "",  # UUID will be assigned later
        "name": "",
        "email": "",
        "skills": entities.get("skills", []),
        "job_titles": entities.get("job_titles", []),
        "years_experience": 0,
        "seniority_level": "",
        "education": [],
        "preferences_raw": preferences_text,
        "target_role": "",
        "target_location": "",
        "salary_expectation_min": None
    }

def parse_cv(file_path: str, preferences_text: str) -> Dict:
    text = extract_text(file_path)
    entities = extract_entities(text)
    user_profile = build_user_profile_json(entities, preferences_text)
    return user_profile