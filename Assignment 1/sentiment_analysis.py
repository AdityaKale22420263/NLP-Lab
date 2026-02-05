import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

# Read the vacation paragraph from file
with open('vacation.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

print("="*60)
print("SENTIMENT ANALYSIS")
print("="*60)

print("\nOriginal Text:")
print(text)

# 1. VADER Sentiment Analysis
print("\n" + "="*60)
print("1. VADER SENTIMENT ANALYZER:")
print("="*60)

sia = SentimentIntensityAnalyzer()

# Overall sentiment
vader_scores = sia.polarity_scores(text)
print("\nOverall Sentiment Scores:")
print(f"Positive: {vader_scores['pos']:.3f}")
print(f"Neutral:  {vader_scores['neu']:.3f}")
print(f"Negative: {vader_scores['neg']:.3f}")
print(f"Compound: {vader_scores['compound']:.3f}")

# Determine overall sentiment
if vader_scores['compound'] >= 0.05:
    overall_sentiment = "POSITIVE"
elif vader_scores['compound'] <= -0.05:
    overall_sentiment = "NEGATIVE"
else:
    overall_sentiment = "NEUTRAL"

print(f"\nOverall Sentiment: {overall_sentiment}")

# Sentence-level sentiment
sentences = sent_tokenize(text)
print("\nSentence-level Sentiment Analysis:")
print("-"*60)
for i, sentence in enumerate(sentences, 1):
    scores = sia.polarity_scores(sentence)
    if scores['compound'] >= 0.05:
        sentiment = "POSITIVE"
    elif scores['compound'] <= -0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    print(f"\nSentence {i}: {sentence}")
    print(f"Sentiment: {sentiment} (Compound: {scores['compound']:.3f})")

# 2. TextBlob Sentiment Analysis
print("\n" + "="*60)
print("2. TEXTBLOB SENTIMENT ANALYZER:")
print("="*60)

blob = TextBlob(text)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

print(f"\nPolarity: {polarity:.3f} (Range: -1 to 1)")
print(f"Subjectivity: {subjectivity:.3f} (Range: 0 to 1)")

if polarity > 0:
    textblob_sentiment = "POSITIVE"
elif polarity < 0:
    textblob_sentiment = "NEGATIVE"
else:
    textblob_sentiment = "NEUTRAL"

print(f"\nOverall Sentiment: {textblob_sentiment}")
print(f"Subjectivity: {'Subjective' if subjectivity > 0.5 else 'Objective'}")

print("\nSentence-level Analysis (TextBlob):")
print("-"*60)
for i, sentence in enumerate(blob.sentences, 1):
    pol = sentence.sentiment.polarity
    if pol > 0:
        sent = "POSITIVE"
    elif pol < 0:
        sent = "NEGATIVE"
    else:
        sent = "NEUTRAL"
    print(f"\nSentence {i}: {sentence}")
    print(f"Sentiment: {sent} (Polarity: {pol:.3f})")


