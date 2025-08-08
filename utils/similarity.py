# utils/similarity.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarities(job_embedding, resume_embeddings):
    # job_embedding: 1D array (shape: [384])
    # resume_embeddings: 2D array (shape: [num_resumes, 384])
    # Returns a 1D numpy array of similarity scores

    # Reshape job_embedding to (1, -1) for pairwise comparison
    job_emb_reshaped = np.array(job_embedding).reshape(1, -1)
    sims = cosine_similarity(job_emb_reshaped, resume_embeddings)
    return sims.flatten()  # shape: (num_resumes,)

def rank_candidates(resume_texts, similarity_scores, top_n=5):
    # resume_texts: list of dicts, each with "name" and "text"
    # similarity_scores: 1D numpy array
    # Returns list of dicts sorted by similarity descending

    indices = np.argsort(-similarity_scores)  # Descending order
    ranked = []
    for idx in indices[:top_n]:
        ranked.append({
            "name": resume_texts[idx]["name"],
            "similarity": float(similarity_scores[idx]),
            "text": resume_texts[idx]["text"]
        })
    return ranked
