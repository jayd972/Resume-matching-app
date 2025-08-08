import os
import pandas as pd
import streamlit as st
import re

from utils.parsing import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
)
from utils.embedding import embed_text, embed_many
from utils.similarity import compute_similarities, rank_candidates
from utils.summary import get_candidate_summary, setup_gemini
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Set your Gemini API key (change as needed)
os.environ["GEMINI_API_KEY"] = "AIzaSyBatxt0LAg_BVaDksQtEI06_6fC8feY9v8"

# --------- Name Extraction Section ---------
def extract_name_regex(text):
    lines = text.strip().split('\n')
    for line in lines[:10]:
        clean = line.strip()
        if 1 < len(clean.split()) <= 4 and all(w and w[0].isupper() for w in clean.split() if w.isalpha()):
            if not any(x in clean.lower() for x in ['resume', 'email', 'phone', 'contact', 'curriculum']):
                return clean
    return "Unknown"

def extract_candidate_name_gemini(text, api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    setup_gemini(key)
    prompt = (
        "Extract ONLY the full name of the candidate from the following resume. "
        "If the name is not present, reply 'Unknown'.\n\n"
        f"Resume:\n{text[:1000]}\n\n"
        "Name:"
    )
    model = genai.GenerativeModel("gemini-2.5-pro")
    try:
        response = model.generate_content(prompt)
        name = response.text.strip()
        name = name.replace('"', '').replace("'", "")
        name = ' '.join(name.split())
        blocklist = [
            "resume", "email", "contact", "phone", "linkedin",
            "skill", "address", "profile", "summary", "curriculum"
        ]
        if not name or "unknown" in name.lower():
            st.warning("Gemini did not return a name.")
            return "Unknown"
        if len(name.split()) > 5 or any(word in name.lower() for word in blocklist):
            st.warning(f"Gemini response looks invalid: {name}")
            return "Unknown"
        return name
    except Exception as e:
        st.error(f"Gemini Exception: {e}")
        return "Unknown"

def extract_candidate_name(text, api_key=None):
    name = extract_candidate_name_gemini(text, api_key)
    if name == "Unknown":
        st.info("Trying regex fallback for candidate name extraction.")
        name = extract_name_regex(text)
    return name

# --------- Streamlit UI Section ---------
st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")
st.title("🕵️‍♂️ Candidate Recommendation Engine")

st.markdown(
    "Paste the job description below and upload candidate resumes. "
    "The app will recommend the best matches with similarity scores and AI-generated fit summaries."
)

# --- JOB DESCRIPTION INPUT ---
job_description = st.text_area(
    "Job Description:",
    height=260,
    placeholder="Paste the full job description here..."
)

# --- UPLOAD FILES ---
uploaded_files = st.file_uploader(
    "Upload candidate resumes (.txt, .pdf, .docx):",
    accept_multiple_files=True,
    type=["txt", "pdf", "docx"]
)

resume_texts = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        ext = name.split(".")[-1].lower()
        try:
            if ext == "txt":
                text = extract_text_from_txt(uploaded_file)
            elif ext == "pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif ext == "docx":
                text = extract_text_from_docx(uploaded_file)
            else:
                text = ""
        except Exception as e:
            text = ""
            st.warning(f"Could not extract text from {name}: {e}")

        if not text.strip():
            st.warning(f"No text could be extracted from {name}.")
        else:
            st.write(f"Extracted text sample from {name}:\n{text[:300]}")

        actual_name = extract_candidate_name(text)
        st.write(f"Detected candidate name: {actual_name}")

        resume_texts.append({"name": actual_name, "text": text})

# --------- Embedding Model Selection ---------
st.markdown("### Select Embedding Model")
model_choice = st.selectbox(
    "Choose the embedding model for similarity computation:",
    [
        "Base SBERT (all-MiniLM-L6-v2)",
        "Fine-Tuned SBERT (model overfitted (Just for the demonstration))",
        "OpenAI (text-embedding-3-small)",
        "Gemini (Google AI Embedding)"  # New!
    ]
)
# Map UI choices to backend code
if model_choice == "Base SBERT (all-MiniLM-L6-v2)":
    embedding_type = "sbert"
    model_name = "all-MiniLM-L6-v2"
elif model_choice == "Fine-Tuned SBERT (model overfitted (Just for the demonstration))":
    embedding_type = "sbert"
    model_name = "Jayd972/fine-tuned-sbert-resume-matcher"
elif model_choice == "OpenAI (text-embedding-3-small)":
    embedding_type = "openai"
    model_name = None
elif model_choice == "Gemini (Google AI Embedding)":
    embedding_type = "gemini"
    model_name = None
else:
    embedding_type = "sbert"
    model_name = "all-MiniLM-L6-v2"

# --- EMBEDDING, SIMILARITY, RANKING, SUMMARY TABLE ---
if job_description and resume_texts:
    try:
        # Embedding
        job_embedding = embed_text(
            job_description,
            model_type=embedding_type,
            model_name=model_name
        )
        resume_text_list = [r["text"] for r in resume_texts]
        resume_embeddings = embed_many(
            resume_text_list,
            model_type=embedding_type,
            model_name=model_name
        )

        # Similarity and ranking
        similarity_scores = compute_similarities(job_embedding, resume_embeddings)
        ranked_candidates = rank_candidates(resume_texts, similarity_scores, top_n=10)

        # Build the table data
        table_data = []
        for candidate in ranked_candidates:
            sim = candidate['similarity']
            verdict = (
                "🟢 Strong fit" if sim >= 0.6 else
                "🟡 Possible fit" if sim >= 0.4 else
                "🔴 Limited fit"
            )
            try:
                summary = get_candidate_summary(
                    job_description,
                    candidate['text'],
                    candidate['similarity']
                )
            except Exception as e:
                summary = f"Error: {e}"

            candidate['verdict'] = verdict
            candidate['summary'] = summary

            table_data.append({
                "Candidate Name": candidate['name'],
                "Similarity Score": f"{sim:.4f}",
                "Verdict": verdict,
                "Summary": summary
            })

        st.markdown("---")
        st.subheader("Top Candidate Recommendations")

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Similarity Score": st.column_config.NumberColumn(format="%.4f", width="small"),
                "Verdict": st.column_config.TextColumn(width="small"),
                "Summary": st.column_config.TextColumn(width="large"),
            }
        )

        # Show full summaries for each candidate below the table
        st.markdown("---")
        st.subheader("Full Candidate Summaries")

        for i, candidate in enumerate(ranked_candidates, start=1):
            st.markdown(f"#### {i}. {candidate['name']}")
            st.markdown(f"**Similarity Score:** `{candidate['similarity']:.4f}`")
            st.markdown(f"**Verdict:** {candidate['verdict']}")
            st.markdown(f"**Summary:** {candidate['summary']}")
            with st.expander("Show extracted resume text"):
                st.code(candidate['text'][:1500] + ("..." if len(candidate['text']) > 1500 else ""))

    except Exception as e:
        st.error(f"Error during processing: {e}")

elif uploaded_files:
    st.info("Please enter a job description to match candidates.")

else:
    st.info("Upload resumes and paste a job description to see recommendations.")

st.caption("🔗 Powered by Streamlit, Sentence Transformers, OpenAI, and Gemini API.")

