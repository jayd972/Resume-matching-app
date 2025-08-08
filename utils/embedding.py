# utils/embedding.py

import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ---------- SBERT ----------
_sbert_models = {}

def get_sbert_model(model_name="all-MiniLM-L6-v2"):
    # Map the demo string to your Hugging Face model
    if model_name == "fine-tuned-sbert-resume-matcher":
        model_name = "Jayd972/fine-tuned-sbert-resume-matcher"
    key = os.path.abspath(model_name) if os.path.isdir(model_name) else model_name
    if key not in _sbert_models:
        print(f"Loading SBERT model from: {model_name}")
        _sbert_models[key] = SentenceTransformer(model_name)
    return _sbert_models[key]


def sbert_embed_text(text, model_name="all-MiniLM-L6-v2"):
    model = get_sbert_model(model_name)
    return model.encode([text])[0]

def sbert_embed_many(texts, model_name="all-MiniLM-L6-v2"):
    model = get_sbert_model(model_name)
    return model.encode(texts)

# ---------- OpenAI ----------
import openai

def openai_embed_text(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding

def openai_embed_many(texts, model="text-embedding-3-small"):
    batch_size = 16
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = [t.replace("\n", " ") for t in texts[i:i+batch_size]]
        response = openai.embeddings.create(
            input=chunk,
            model=model
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings

# ---------- Gemini (Google AI) ----------
def gemini_embed_text(text, model="models/embedding-001"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    response = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

def gemini_embed_many(texts, model="models/embedding-001"):
    return [gemini_embed_text(t, model=model) for t in texts]

# ---------- Unified Embedding Selector ----------
def embed_text(text, model_type="sbert", model_name="all-MiniLM-L6-v2"):
    if model_type == "sbert":
        return sbert_embed_text(text, model_name=model_name)
    elif model_type == "openai":
        return openai_embed_text(text)
    elif model_type == "gemini":
        return gemini_embed_text(text)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def embed_many(texts, model_type="sbert", model_name="all-MiniLM-L6-v2"):
    if model_type == "sbert":
        return sbert_embed_many(texts, model_name=model_name)
    elif model_type == "openai":
        return openai_embed_many(texts)
    elif model_type == "gemini":
        return gemini_embed_many(texts)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

