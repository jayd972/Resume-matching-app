# utils/summary.py
import google.generativeai as genai
import os

def setup_gemini(api_key=None):
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

def get_candidate_summary(job_description, resume_text, similarity_score, api_key=None):
    setup_gemini(api_key)
    prompt = (
        "Given the following job description and candidate resume, "
        "write 2-3 sentences explaining how well the candidate fits the role. "
        "If the similarity score (between 0 and 1) is below 0.5, politely highlight potential gaps or missing experience.\n\n"
        f"Similarity Score: {similarity_score:.2f}\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate Resume:\n{resume_text[:2500]}\n\n"
        "Fit Assessment:"
    )
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()
