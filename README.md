# 🕵️‍♂️ Candidate Recommendation Engine

A web app that recommends the best candidates for a job, based on resume relevance.  
Built with **Streamlit** and supports multiple embedding models (SBERT, OpenAI, Google Gemini).

---

## ✨ Features

- Paste a **job description**
- Upload multiple resumes (`.txt`, `.pdf`, `.docx`)
- **Automatic name extraction** (with Gemini AI fallback)
- Choose between different **embedding models**:
  - Base SBERT (`all-MiniLM-L6-v2`)
  - Fine-tuned SBERT (demo model)
  - OpenAI (`text-embedding-3-small`)
  - Gemini (Google AI Embedding)
- Compute **cosine similarity** between job and each resume
- **Top 5–10 recommended candidates** with:
  - Name
  - Similarity score
  - Verdict (Strong/Possible/Limited fit)
  - **AI-generated summary** (Gemini 2.5 Pro)
- View resume snippets and full text
- **Download results as an Excel file** (`.xlsx`) including all names, scores, and summaries

---

## 🖥️ Usage

1. Paste the **job description** in the provided box.
2. Upload **candidate resumes** (any combination of `.txt`, `.pdf`, or `.docx`).
3. Select the **embedding model** of your choice.
4. Click **Run** (automatically updates).
5. See the **top recommended candidates**, their similarity scores, and AI summaries.
6. **Download all results as an Excel file** with one click.

---


---

## 🛠️ Assumptions & Limitations

- **Resume parsing:** Only `.txt`, `.pdf`, and `.docx` supported.  
- **Name extraction:** Uses AI (Gemini) and regex fallback — may not be perfect for all formats.
- **API keys required** for OpenAI and Gemini models.
- **PDF parsing** relies on text layer, so scanned images (non-OCR) may not extract well.
- **Similarity scores** depend on embedding quality; try different models if results seem off.
- **AI summary** uses up to 2,500 characters of resume for cost/speed reasons.
- **Excel download:** All recommendations are included in a ready-to-use `.xlsx` file.

---

## 👤 Author

Built by Jay Darji  
_Contact: [darjijay972@gmail.com](mailto:darjijay972@gmail.com)_

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Sentence Transformers (SBERT)](https://www.sbert.net/)
- [OpenAI API](https://platform.openai.com/)
- [Google Gemini](https://ai.google.dev/gemini-api/docs)

---


## 📂 File Structure

