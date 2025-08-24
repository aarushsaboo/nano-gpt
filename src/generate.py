import torch
import os
from model import NanoGPT
from tokenizer import CharacterTokenizer

class TextGenerator:
    def __init__(self, model_path='models/nano_gpt.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        
        # Load the saved data
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Recreate tokenizer
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.char_to_idx = checkpoint['char_to_idx']
        self.tokenizer.idx_to_char = checkpoint['idx_to_char']
        self.tokenizer.vocab_size = checkpoint['vocab_size']
        
        # Recreate model
        config = checkpoint['model_config']
        self.model = NanoGPT(
            vocab_size=checkpoint['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            max_seq_len=config['max_seq_len']
        )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {len(self.tokenizer)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate(self, prompt="To be", max_length=200, temperature=1.0, top_k=None):
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        print(f"Generating text with prompt: '{prompt}'")
        
        # Encode the prompt
        tokens = self.tokenizer.encode(prompt)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Convert to tensor and move to device
                input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                
                # Get model predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, indices, values)
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                tokens.append(next_token)
                
                # Optional: stop at sentence endings
                next_char = self.tokenizer.decode([next_token])
                if len(tokens) > len(self.tokenizer.encode(prompt)) + 10 and next_char in '.!?\n':
                    break
        
        generated_text = self.tokenizer.decode(tokens)
        return generated_text
    
    def interactive_generation(self):
        print("\n=== Interactive Text Generation ===")
        print("Enter prompts to generate Shakespeare-like text!")
        print("Commands: 'quit' to exit, 'help' for options")
        
        while True:
            try:
                user_input = input("\nEnter prompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("Options:")
                    print("  - Enter any text prompt")
                    print("  - 'temp X' to set temperature (e.g., 'temp 0.8')")
                    print("  - 'len X' to set max length (e.g., 'len 100')")
                    continue
                
                if user_input.startswith('temp '):
                    try:
                        temp = float(user_input.split()[1])
                        self.temperature = temp
                        print(f"Temperature set to {temp}")
                        continue
                    except:
                        print("Invalid temperature format. Use 'temp 0.8'")
                        continue
                
                if user_input.startswith('len '):
                    try:
                        length = int(user_input.split()[1])
                        self.max_length = length
                        print(f"Max length set to {length}")
                        continue
                    except:
                        print("Invalid length format. Use 'len 100'")
                        continue
                
                if not user_input:
                    user_input = "To be"
                
                # Generate text
                generated = self.generate(
                    prompt=user_input,
                    max_length=getattr(self, 'max_length', 100),
                    temperature=getattr(self, 'temperature', 1.0)
                )
                
                print(f"\nGenerated text:")
                print(f"'{generated}'")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Shakespeare Text Generator ===")
    
    try:
        # Create generator
        generator = TextGenerator()
        
        # Test some prompts
        test_prompts = [
            "To be or not to be",
            "What light through yonder", 
            "Romeo",
            "The quick brown"
        ]
        
        print("\n=== Sample Generations ===")
        for prompt in test_prompts:
            print(f"\nPrompt: '{prompt}'")
            generated = generator.generate(prompt, max_length=80, temperature=0.8)
            print(f"Generated: '{generated}'")
            print("-" * 50)
        
        # Start interactive mode
        generator.interactive_generation()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've trained a model first by running: python src/trainer.py")
    except Exception as e:
        print(f"Unexpected error: {e}")