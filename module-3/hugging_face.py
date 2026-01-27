from transformers import AutoTokenizer, AutoModel
import torch

# Model hosted on Hugging Face Hub
MODEL_NAME = "distilbert-base-uncased"

def main():
    # Load tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Sample input text
    text = "Hugging Face Hub makes model sharing easy."

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Access model output
    last_hidden_state = outputs.last_hidden_state

    print("Input text:", text)
    print("Tokenized input keys:", inputs.keys())
    print("Output shape:", last_hidden_state.shape)

if __name__ == "__main__":
    main()
