# utils/parsing.py

import pdfplumber
from docx import Document

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    # Save the uploaded file to a temporary location
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        document = Document(tmp.name)
        text = "\n".join([para.text for para in document.paragraphs])
    return text
