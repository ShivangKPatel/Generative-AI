from transformers import AutoTokenizer
import re
import sentencepiece as spm

# Example 1: Using Hugging Face Transformers (BPE - Byte Pair Encoding)
def tokenize_with_huggingface():
    """Tokenize text using a pre-trained tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    text = "Tokenization is important for NLP!"
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print("Text:", text)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    return tokens, token_ids


# Example 2: Manual subword tokenization (WordPiece-like)
def manual_subword_tokenization(text, vocab_size=1000):
    """Simple character-level to subword tokenization"""
    # Split by spaces and punctuation
    words = re.findall(r'\b\w+\b|\W', text.lower())
    
    tokens = []
    for word in words:
        if len(word) > 1 and word.isalpha():
            # Add ## prefix for subword tokens (except first)
            tokens.append(word[:3])  # Simplified: take first 3 chars
        else:
            tokens.append(word)
    
    return tokens


# Example 3: Using SentencePiece (unigram tokenization)
def tokenize_with_sentencepiece():
    """Example of SentencePiece tokenization"""
    try:
        
        # Load pre-trained model
        sp = spm.SentencePieceProcessor()
        sp.load('model.model')
        
        text = "Tokenization is important"
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        
        print("Pieces:", pieces)
        print("IDs:", ids)
    except ImportError:
        print("SentencePiece not installed")


if __name__ == "__main__":
    # Run HuggingFace tokenizer
    print("=== Hugging Face Tokenizer ===")
    tokenize_with_huggingface()
    
    print("\n=== Manual Subword Tokenization ===")
    text = "Understanding tokenization in NLP"
    tokens = manual_subword_tokenization(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")