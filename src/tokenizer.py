class CharacterTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def build_vocabulary(self, text):
        unique_chars = sorted(set(text)) # find all unique characters, sort them & assign numbers
        self.vocab_size = len(unique_chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        print(f"Vocabulary built! Found {self.vocab_size} unique characters")
        
    def encode(self, text):
        return [self.char_to_idx[char] for char in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def __len__(self):
        return self.vocab_size


if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    sample_text = "Hello world! This is a test."
    
    tokenizer.build_vocabulary(sample_text)
    
    encoded = tokenizer.encode("Hello")
    print(f"'Hello' encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded back: '{decoded}'")
    
    print("First 10 vocabulary mappings:")
    for i in range(min(10, len(tokenizer.idx_to_char))):
        print(f"  {i} -> '{tokenizer.idx_to_char[i]}'")