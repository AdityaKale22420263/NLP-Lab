import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Read the vacation paragraph from file
with open('vacation.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

print("="*60)
print("TOKENIZATION")
print("="*60)

print("\nOriginal Text:")
print(text)

# Sentence Tokenization
sentences = sent_tokenize(text)
print("\n" + "="*60)
print("1. SENTENCE TOKENIZATION:")
print("="*60)
print(f"Total sentences: {len(sentences)}\n")
for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")

# Word Tokenization
words = word_tokenize(text)
print("\n" + "="*60)
print("2. WORD TOKENIZATION:")
print("="*60)
print(f"Total words/tokens: {len(words)}\n")
print("First 20 tokens:", words[:20])
print("\nAll tokens:", words)