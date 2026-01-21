import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import requests
import io
from transformers import AutoTokenizer, AutoModel
import torch

import matplotlib.pyplot as plt

print("=" * 50)
print("1. WORD2VEC EMBEDDINGS")
print("=" * 50)

# Sample sentences for training
sentences = [
    "the quick brown fox jumps over the lazy dog".split(),
    "a fast brown fox leaps over a sleepy dog".split(),
    "the dog sat on the mat".split(),
    "the cat sat on the mat".split(),
]

# Train Word2Vec model
w2v_model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, workers=4)
print(f"Word2Vec embedding for 'fox': {w2v_model.wv['fox'][:5]}...")
print(f"Similarity between 'fox' and 'dog': {w2v_model.wv.similarity('fox', 'dog'):.4f}")

print("\n" + "=" * 50)
print("2. CONTEXTUAL EMBEDDINGS (BERT/Transformers)")
print("=" * 50)

try:
    
    # Load pre-trained model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Example sentences
    sentences_contextual = [
        "The bank can provide loans.",
        "I sat by the bank of the river."
    ]
    
    print(f"\nContextual embeddings for the word 'bank':")
    print("(Notice how 'bank' has different representations based on context)\n")
    
    for sentence in sentences_contextual:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state
        
        # Find the embedding of the word "bank"
        bank_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("bank"))
        if bank_token_idx:
            bank_embedding = embeddings[0, 1, :5].numpy()  # First 5 dimensions
            print(f"Sentence: '{sentence}'")
            print(f"'bank' embedding (first 5 dims): {bank_embedding}...\n")
            
except ImportError:
    print("Transformers library not installed. Install with: pip install transformers torch")

print("\n" + "=" * 50)
print("3. EMBEDDING COMPARISON")
print("=" * 50)

print("\nWord2Vec properties:")
print(f"  - Static embeddings (same representation regardless of context)")
print(f"  - Trained on: {len(sentences)} sentences")
print(f"  - Vocabulary size: {len(w2v_model.wv)}")
print(f"  - Vector dimensions: {w2v_model.wv.vector_size}")

print("\nContextual Embeddings properties:")
print(f"  - Dynamic embeddings (vary based on context)")
print(f"  - Trained on large corpora (e.g., Wikipedia, Books)")
print(f"  - Better for: NER, sentiment analysis, Q&A systems")