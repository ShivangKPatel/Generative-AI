import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PrefixTuningConfig,
    PromptTuningConfig,
    PeftModel,
)
import google.generativeai as genai
from tqdm import tqdm
import os
from typing import List, Dict


# ========== 1. CUSTOM DATASET CLASS ==========

class TextDataset(Dataset):
    """Custom dataset for fine-tuning"""
    
    def __init__(self, texts: List[str], labels: List[int] = None, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        # For causal LM, labels are the same as input_ids
        item['labels'] = encoding['input_ids'].squeeze()
        
        # For classification tasks, add label
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx])
        
        return item


# ========== 2. SAMPLE DATA ==========

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This is a sample sentence for fine-tuning.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn.",
    "Fine-tuning helps adapt pre-trained models to specific downstream tasks effectively.",
    "Deep learning models require large amounts of data to achieve optimal performance.",
    "Transfer learning allows us to leverage knowledge from one task to improve another.",
    "Natural language processing has revolutionized how computers understand human language.",
    "Transformer models have become the foundation of modern NLP applications worldwide.",
    "Neural networks are inspired by the biological neural systems found in animal brains.",
]


# ========== 3. FULL FINE-TUNING ==========

class FullFineTuning:
    """Full Fine-Tuning: Update all model parameters"""
    
    def __init__(self, model_name: str = "gpt2", learning_rate: float = 5e-5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = self.total_params
        
    def train(self, train_texts: List[str], epochs: int = 3, batch_size: int = 2):
        """Train the model with full fine-tuning"""
        print("\n" + "="*60)
        print("FULL FINE-TUNING")
        print("="*60)
        print(f"Total parameters: {self.total_params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")
        print(f"Training on device: {self.device}\n")
        
        # Create dataset and dataloader
        dataset = TextDataset(train_texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            
            for batch in tqdm(dataloader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss: {avg_loss:.4f}\n")
        
        print("Full Fine-Tuning completed!\n")
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


# ========== 4. LoRA FINE-TUNING ==========

class LoRAFineTuning:
    """LoRA (Low-Rank Adaptation): Update low-rank decomposition matrices"""
    
    def __init__(self, model_name: str = "gpt2", learning_rate: float = 1e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,                          # Rank of LoRA matrices
            lora_alpha=16,                # Scaling factor
            target_modules=["c_attn"],    # Target attention modules
            lora_dropout=0.1,             # Dropout for LoRA layers
            bias="none",                  # No bias for LoRA
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self, train_texts: List[str], epochs: int = 3, batch_size: int = 2):
        """Train the model with LoRA"""
        print("\n" + "="*60)
        print("LoRA FINE-TUNING")
        print("="*60)
        print(f"Total parameters: {self.total_params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")
        print(f"Parameter reduction: {(1 - self.trainable_params/self.total_params)*100:.2f}%")
        print(f"Training on device: {self.device}\n")
        
        # Create dataset and dataloader
        dataset = TextDataset(train_texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            
            for batch in tqdm(dataloader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss: {avg_loss:.4f}\n")
        
        print("LoRA Fine-Tuning completed!\n")
    
    def save_model(self, save_path: str):
        """Save the LoRA model"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"LoRA model saved to {save_path}")


# ========== 5. PEFT (PREFIX TUNING) ==========

class PrefixTuningFineTuning:
    """Prefix Tuning: Add trainable prefix vectors to the input"""
    
    def __init__(self, model_name: str = "gpt2", learning_rate: float = 1e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = learning_rate
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Configure Prefix Tuning
        prefix_config = PrefixTuningConfig(
            num_virtual_tokens=20,        # Number of virtual tokens
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply Prefix Tuning to model
        self.model = get_peft_model(self.model, prefix_config)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self, train_texts: List[str], epochs: int = 3, batch_size: int = 2):
        """Train the model with Prefix Tuning"""
        print("\n" + "="*60)
        print("PREFIX TUNING (PEFT)")
        print("="*60)
        print(f"Total parameters: {self.total_params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")
        print(f"Parameter reduction: {(1 - self.trainable_params/self.total_params)*100:.2f}%")
        print(f"Training on device: {self.device}\n")
        
        # Create dataset and dataloader
        dataset = TextDataset(train_texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            
            for batch in tqdm(dataloader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss: {avg_loss:.4f}\n")
        
        print("Prefix Tuning completed!\n")
    
    def save_model(self, save_path: str):
        """Save the prefix tuning model"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Prefix Tuning model saved to {save_path}")


# ========== 6. GEMINI MODEL WITH FINE-TUNING ==========

class GeminiFineTuning:
    """Fine-Tune Gemini Model using Google's API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini model
        
        Args:
            api_key: Google API key (can also be set via GOOGLE_API_KEY env var)
        """
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            raise ValueError("Please provide API key or set GOOGLE_API_KEY environment variable")
        
        self.model_name = "gemini-1.5-flash"
        print(f"\nInitialized Gemini model: {self.model_name}")
    
    def prepare_training_data(self, texts: List[str], labels: List[str] = None) -> List[Dict]:
        """
        Prepare training data for Gemini fine-tuning
        
        Args:
            texts: List of training texts
            labels: List of labels/outputs
        
        Returns:
            Formatted training data
        """
        training_data = []
        
        for i, text in enumerate(texts):
            # For fine-tuning, we need question-answer pairs or input-output pairs
            training_data.append({
                "role": "user",
                "parts": [{"text": text}],
            })
            
            # Add a response (you should customize this)
            if labels:
                response = labels[i]
            else:
                response = f"Learned response for: {text[:50]}..."
            
            training_data.append({
                "role": "model",
                "parts": [{"text": response}],
            })
        
        return training_data
    
    def fine_tune(self, training_data: List[Dict], epochs: int = 1):
        """
        Fine-tune Gemini model
        
        Note: Actual fine-tuning requires using Vertex AI or the official API
        This demonstrates the API structure
        """
        print("\n" + "="*60)
        print("GEMINI FINE-TUNING")
        print("="*60)
        print("Note: Full fine-tuning of Gemini requires Vertex AI access\n")
        
        # For demonstration: Create a tuning job
        training_data_formatted = self.prepare_training_data(SAMPLE_TEXTS)
        
        print(f"Prepared {len(training_data_formatted)} training samples")
        print("To fine-tune Gemini, use Google Cloud's Vertex AI")
        print("Or access through the official Google AI API\n")
    
    def generate_with_context(self, prompt: str, context: str = None):
        """Generate text using Gemini with optional context"""
        print("\n" + "="*60)
        print("GEMINI TEXT GENERATION")
        print("="*60)
        
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(full_prompt)
            print(f"Prompt: {prompt}")
            print(f"Response: {response.text}\n")
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            print("Make sure your API key is set correctly\n")
            return None


# ========== 7. COMPARISON UTILITY ==========

class FinetuningComparison:
    """Compare different fine-tuning methods"""
    
    @staticmethod
    def compare_methods():
        """Display comparison of fine-tuning methods"""
        print("\n" + "="*70)
        print("FINE-TUNING METHODS COMPARISON")
        print("="*70 + "\n")
        
        comparison_data = {
            "Method": ["Full Fine-Tuning", "LoRA", "Prefix Tuning", "Prompt Tuning"],
            "Parameters Updated": ["All", "Low-rank (2-5%)", "Prefix vectors (1-2%)", "Prompt embeddings (0.1%)"],
            "Memory Usage": ["High", "Low", "Very Low", "Very Low"],
            "Speed": ["Slow", "Fast", "Fast", "Very Fast"],
            "Performance": ["Best", "Excellent", "Good", "Fair"],
            "Task Adaptation": ["Excellent", "Excellent", "Good", "Fair"],
        }
        
        # Print comparison table
        col_widths = [20, 25, 20, 20, 20, 20]
        headers = list(comparison_data.keys())
        
        print(f"{'Method':<20} {'Parameters Updated':<25} {'Memory':<20} {'Speed':<20} {'Performance':<20} {'Adaptation':<20}")
        print("-" * 125)
        
        for i in range(len(comparison_data["Method"])):
            print(f"{comparison_data['Method'][i]:<20} {comparison_data['Parameters Updated'][i]:<25} {comparison_data['Memory Usage'][i]:<20} {comparison_data['Speed'][i]:<20} {comparison_data['Performance'][i]:<20} {comparison_data['Task Adaptation'][i]:<20}")
        
        print("\n" + "="*70 + "\n")


# ========== 8. TESTING MODULE ==========

class ModelTester:
    """Test and evaluate fine-tuned models"""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, is_peft_model: bool = False):
        """
        Initialize tester with model
        
        Args:
            model_path: Path to fine-tuned model
            tokenizer_path: Path to tokenizer (same as model_path if None)
            is_peft_model: Whether this is a PEFT model (LoRA, Prefix Tuning, etc.)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_path = tokenizer_path or model_path
        self.is_peft = is_peft_model
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if is_peft_model:
                # For PEFT models, load the base model first, then the adapter
                base_model = AutoModelForCausalLM.from_pretrained(model_path)
                self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
            else:
                # For regular models, load directly
                self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
            
            self.model.eval()
            self.model_name = model_path.split('/')[-1]
            print(f"✓ Loaded model from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, 
                     top_p: float = 0.9, num_return_sequences: int = 1) -> List[str]:
        """
        Generate text using the fine-tuned model
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
        
        Returns:
            List of generated texts
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                no_repeat_ngram_size=2
            )
        
        # Decode outputs
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def evaluate_on_test_set(self, test_texts: List[str], batch_size: int = 2) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            test_texts: List of test texts
            batch_size: Batch size for evaluation
        
        Returns:
            Evaluation metrics
        """
        dataset = TextDataset(test_texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        total_loss = 0
        num_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'average_loss': avg_loss,
            'num_samples': len(test_texts),
            'num_batches': num_batches
        }
    
    def get_perplexity(self, texts: List[str]) -> float:
        """
        Calculate perplexity on texts
        
        Args:
            texts: List of texts
        
        Returns:
            Perplexity value
        """
        dataset = TextDataset(texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1)
        
        total_loss = 0
        num_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item() * input_ids.shape[1]
                num_tokens += input_ids.shape[1]
        
        perplexity = torch.exp(torch.tensor(total_loss / num_tokens)).item()
        return perplexity
    
    def test_multiple_prompts(self, prompts: List[str], max_length: int = 80) -> Dict:
        """
        Test model with multiple prompts
        
        Args:
            prompts: List of prompts to test
            max_length: Maximum generation length
        
        Returns:
            Dictionary with results
        """
        results = {}
        
        for i, prompt in enumerate(prompts, 1):
            generated = self.generate_text(prompt, max_length=max_length)
            results[f"prompt_{i}"] = {
                'input': prompt,
                'outputs': generated
            }
        
        return results
    
    def compare_models(self, other_model_path: str, test_prompts: List[str]) -> Dict:
        """
        Compare this model with another model
        
        Args:
            other_model_path: Path to other model
            test_prompts: Prompts to test both models on
        
        Returns:
            Comparison results
        """
        other_tester = ModelTester(other_model_path)
        
        comparison = {
            'model_1': self.model_name,
            'model_2': other_tester.model_name,
            'comparisons': []
        }
        
        for prompt in test_prompts:
            output_1 = self.generate_text(prompt)[0]
            output_2 = other_tester.generate_text(prompt)[0]
            
            comparison['comparisons'].append({
                'prompt': prompt,
                self.model_name: output_1,
                other_tester.model_name: output_2
            })
        
        return comparison
    
    def print_test_report(self, test_data: Dict):
        """Print formatted test report"""
        print("\n" + "="*70)
        print(f"TEST REPORT: {self.model_name}")
        print("="*70)
        print(f"Average Loss: {test_data['average_loss']:.4f}")
        print(f"Number of Samples: {test_data['num_samples']}")
        print(f"Number of Batches: {test_data['num_batches']}")
        print("="*70 + "\n")


# ========== 9. COMPREHENSIVE TESTING FUNCTION ==========

def run_model_tests():
    """Comprehensive testing of all fine-tuned models"""
    
    # Test prompts
    TEST_PROMPTS = [
        "Machine learning is",
        "The future of AI",
        "Deep learning models",
        "Fine-tuning helps with",
        "Natural language processing is"
    ]
    
    TEST_TEXTS = [
        "This is a test sentence for evaluation.",
        "Testing the fine-tuned model on new data.",
        "Neural networks are powerful tools.",
        "Transfer learning is effective.",
        "Data is essential for training models."
    ]
    
    print("\n" + "="*70)
    print("FINE-TUNED MODEL TESTING")
    print("="*70)
    
    # Test Full Fine-Tuning Model
    if os.path.exists("./fine_tuned_models/full_ft_model"):
        print("\n### Testing Full Fine-Tuning Model ###\n")
        full_tester = ModelTester("./fine_tuned_models/full_ft_model")
        
        # Generate text
        print("Generated Samples:")
        for prompt in TEST_PROMPTS[:2]:
            outputs = full_tester.generate_text(prompt, max_length=80)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {outputs[0]}")
        
        # Evaluate
        print("\n\nEvaluation on Test Set:")
        eval_results = full_tester.evaluate_on_test_set(TEST_TEXTS)
        full_tester.print_test_report(eval_results)
        
        # Perplexity
        print("Calculating Perplexity...")
        perplexity = full_tester.get_perplexity(TEST_TEXTS)
        print(f"Perplexity: {perplexity:.4f}\n")
    
    # Test LoRA Model
    if os.path.exists("./fine_tuned_models/lora_model"):
        print("\n### Testing LoRA Model ###\n")
        lora_tester = ModelTester("./fine_tuned_models/lora_model", is_peft_model=True)
        
        # Generate text
        print("Generated Samples:")
        for prompt in TEST_PROMPTS[:2]:
            outputs = lora_tester.generate_text(prompt, max_length=80)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {outputs[0]}")
        
        # Evaluate
        print("\n\nEvaluation on Test Set:")
        eval_results = lora_tester.evaluate_on_test_set(TEST_TEXTS)
        lora_tester.print_test_report(eval_results)
        
        # Perplexity
        print("Calculating Perplexity...")
        perplexity = lora_tester.get_perplexity(TEST_TEXTS)
        print(f"Perplexity: {perplexity:.4f}\n")
    
    # Test Prefix Tuning Model
    if os.path.exists("./fine_tuned_models/prefix_tuning_model"):
        print("\n### Testing Prefix Tuning Model ###\n")
        prefix_tester = ModelTester("./fine_tuned_models/prefix_tuning_model", is_peft_model=True)
        
        # Generate text
        print("Generated Samples:")
        for prompt in TEST_PROMPTS[:2]:
            outputs = prefix_tester.generate_text(prompt, max_length=80)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {outputs[0]}")
        
        # Evaluate
        print("\n\nEvaluation on Test Set:")
        eval_results = prefix_tester.evaluate_on_test_set(TEST_TEXTS)
        prefix_tester.print_test_report(eval_results)
        
        # Perplexity
        print("Calculating Perplexity...")
        perplexity = prefix_tester.get_perplexity(TEST_TEXTS)
        print(f"Perplexity: {perplexity:.4f}\n")
    
    # Comparative Testing
    if os.path.exists("./fine_tuned_models/full_ft_model") and os.path.exists("./fine_tuned_models/lora_model"):
        print("\n### Comparative Testing: Full Fine-Tuning vs LoRA ###\n")
        comparison = full_tester.compare_models(
            "./fine_tuned_models/lora_model",
            TEST_PROMPTS[:3]
        )
        
        for comp in comparison['comparisons']:
            print(f"\nPrompt: {comp['prompt']}")
            print(f"Full FT: {comp['full_ft_model'][:80]}...")
            print(f"LoRA:    {comp['lora_model'][:80]}...")
    
    print("\n" + "="*70)
    print("TESTING COMPLETED!")
    print("="*70 + "\n")


# ========== 8. MAIN EXECUTION ==========

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE FINE-TUNING GUIDE")
    print("Full Fine-Tuning vs LoRA vs PEFT vs Gemini")
    print("="*70)
    
    # Configuration
    EPOCHS = 2
    BATCH_SIZE = 2
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 1. Full Fine-Tuning
    print("\n### Starting Full Fine-Tuning ###")
    full_ft = FullFineTuning(model_name="gpt2", learning_rate=5e-5)
    full_ft.train(SAMPLE_TEXTS, epochs=EPOCHS, batch_size=BATCH_SIZE)
    full_ft.save_model("./fine_tuned_models/full_ft_model")
    
    # 2. LoRA Fine-Tuning
    print("\n### Starting LoRA Fine-Tuning ###")
    lora_ft = LoRAFineTuning(model_name="gpt2", learning_rate=1e-4)
    lora_ft.train(SAMPLE_TEXTS, epochs=EPOCHS, batch_size=BATCH_SIZE)
    lora_ft.save_model("./fine_tuned_models/lora_model")
    
    # 3. Prefix Tuning (PEFT)
    print("\n### Starting Prefix Tuning ###")
    prefix_ft = PrefixTuningFineTuning(model_name="gpt2", learning_rate=1e-4)
    prefix_ft.train(SAMPLE_TEXTS, epochs=EPOCHS, batch_size=BATCH_SIZE)
    prefix_ft.save_model("./fine_tuned_models/prefix_tuning_model")
    
    # 4. Gemini Fine-Tuning (if API key available)
    print("\n### Gemini Fine-Tuning Info ###")
    try:
        gemini_ft = GeminiFineTuning(api_key="AIzaSyDYS-HWKO7zulzdS1b9JVjjLG9I3DtQjhY")
        gemini_ft.fine_tune(SAMPLE_TEXTS)
        # Uncomment below if you have API access and want to generate
        gemini_ft.generate_with_context("What is machine learning?")
    except ValueError as e:
        print(f"Gemini setup skipped: {e}")
    except Exception as e:
        print(f"Note: {e}")
    
    # 5. Display Comparison
    FinetuningComparison.compare_methods()
    
    print("="*70)
    print("ALL FINE-TUNING DEMONSTRATIONS COMPLETED!")
    print("="*70)
    
    # 6. Test Fine-Tuned Models
    print("\n### Testing Fine-Tuned Models ###")
    run_model_tests()


if __name__ == "__main__":
    main()
