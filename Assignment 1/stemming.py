import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Download required NLTK data
nltk.download('punkt')

# Read the vacation paragraph from file
with open('vacation.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

print("="*60)
print("STEMMING")
print("="*60)

print("\nOriginal Text:")
print(text)

# Initialize stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

# Tokenize
words = word_tokenize(text)

# Apply different stemming algorithms
porter_stemmed = [porter.stem(word) for word in words]
lancaster_stemmed = [lancaster.stem(word) for word in words]
snowball_stemmed = [snowball.stem(word) for word in words]

print("\n" + "="*60)
print("1. PORTER STEMMER:")
print("="*60)
print(f"{'Original':<25} {'Stemmed':<25}")
print("-"*50)
for word, stem in zip(words, porter_stemmed):
    if word.lower() != stem:
        print(f"{word:<25} {stem:<25}")

print("\n" + "="*60)
print("2. LANCASTER STEMMER:")
print("="*60)
print(f"{'Original':<25} {'Stemmed':<25}")
print("-"*50)
for word, stem in zip(words, lancaster_stemmed):
    if word.lower() != stem:
        print(f"{word:<25} {stem:<25}")

print("\n" + "="*60)
print("3. SNOWBALL STEMMER:")
print("="*60)
print(f"{'Original':<25} {'Stemmed':<25}")
print("-"*50)
for word, stem in zip(words, snowball_stemmed):
    if word.lower() != stem:
        print(f"{word:<25} {stem:<25}")

print("\n" + "="*60)
print("Complete Stemmed Text (Porter):")
print("="*60)
print(" ".join(porter_stemmed))