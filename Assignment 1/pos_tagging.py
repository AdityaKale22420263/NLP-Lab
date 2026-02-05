import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Read the vacation paragraph from file
with open('vacation.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

print("="*60)
print("PART-OF-SPEECH (POS) TAGGING")
print("="*60)

print("\nOriginal Text:")
print(text)

# Tokenize
words = word_tokenize(text)

# POS Tagging
pos_tags = pos_tag(words)

print("\n" + "="*60)
print("POS Tags for all tokens:")
print("="*60)
print(f"\n{'Word':<20} {'POS Tag':<15} {'Description':<30}")
print("-"*65)

# Dictionary for common POS tag descriptions
pos_descriptions = {
    'NN': 'Noun, singular',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund/present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'DT': 'Determiner',
    'IN': 'Preposition/subordinating conjunction',
    'CC': 'Coordinating conjunction',
    'TO': 'to',
    'CD': 'Cardinal number',
    'WP': 'Wh-pronoun',
    'WDT': 'Wh-determiner',
    'WRB': 'Wh-adverb',
    '.': 'Punctuation mark',
    ',': 'Comma',
}

for word, pos in pos_tags:
    description = pos_descriptions.get(pos, 'Other')
    print(f"{word:<20} {pos:<15} {description:<30}")

# Count POS tags
from collections import Counter
pos_counts = Counter([pos for word, pos in pos_tags])

print("\n" + "="*60)
print("POS Tag Frequency:")
print("="*60)
for pos, count in pos_counts.most_common():
    description = pos_descriptions.get(pos, 'Other')
    print(f"{pos:<10} {count:<5} - {description}")