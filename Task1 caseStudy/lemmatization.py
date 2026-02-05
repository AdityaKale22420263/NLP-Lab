import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Read the vacation paragraph from file
with open('vacation.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

print("="*60)
print("LEMMATIZATION")
print("="*60)

print("\nOriginal Text:")
print(text)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenize
words = word_tokenize(text)

# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Get POS tags for better lemmatization
pos_tags = nltk.pos_tag(words)

# Lemmatize with POS tags
lemmatized_words = []
for word, pos in pos_tags:
    wordnet_pos = get_wordnet_pos(pos)
    lemma = lemmatizer.lemmatize(word.lower(), pos=wordnet_pos)
    lemmatized_words.append(lemma)

print("\n" + "="*60)
print("Original vs Lemmatized (showing differences):")
print("="*60)
print(f"\n{'Original':<20} {'Lemmatized':<20} {'POS':<10}")
print("-"*50)

for i, (word, lemma, (_, pos)) in enumerate(zip(words, lemmatized_words, pos_tags)):
    if word.lower() != lemma:
        print(f"{word:<20} {lemma:<20} {pos:<10}")

print("\n" + "="*60)
print("Complete Lemmatized Text:")
print("="*60)
print(" ".join(lemmatized_words))