# NLP Lab Assignments

A structured collection of Natural Language Processing assignments implemented using Python, NLTK, scikit-learn, and related NLP libraries.

This repository follows a consistent folder-by-assignment format, where each assignment contains all relevant code and generated outputs.

## Repository Structure

```
NLP/
	Assignment 1/
	Assignment 2/
	Assignment 3/
	Assignment 4/
	Assignment 5/
	Assignment 6/
	Assignment7/
	Assignment 8/
	Assignment 9/
	Assignment 10/
```

## Assignment Index

| Assignment | Primary File(s) | Focus Area |
|---|---|---|
| 1 | `tokenization.py`, `stemming.py`, `lemmatization.py`, `pos_tagging.py`, `sentiment_analysis.py`, `combined.py` | Core NLP preprocessing and baseline analysis |
| 2 | `code.py` | BOW, TF-IDF, Word2Vec, similarity and visualization |
| 3 | `code.py` | Advanced preprocessing pipeline and TF-IDF features |
| 4 | `code.py` | Named Entity Recognition and evaluation |
| 5 | `code.py` | NLP practical implementation (script-based) |
| 6 | `code.ipynb` | Notebook-based assignment |
| 7 | `code.ipynb` | Notebook-based assignment |
| 8 | `code.py` | Word Sense Disambiguation using WordNet-based methods |
| 9 | `code.py` | Indian language sentiment analysis with translation-assisted pipeline |
| 10 | `code.py` | N-gram language modeling for autocomplete and text generation |

## Requirements

- Python 3.9+
- pip (latest recommended)

Common libraries used across assignments:

- nltk
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- gensim
- spacy
- textblob
- deep-translator
- vaderSentiment

Install dependencies:

```bash
pip install nltk numpy pandas matplotlib seaborn scikit-learn gensim spacy textblob deep-translator vaderSentiment
python -m spacy download en_core_web_sm
```

## Setup

1. Open a terminal in the `NLP` folder.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies listed above.
4. Run assignments using either Python scripts or Jupyter notebooks.

## How To Run

### Script-based assignments

From the `NLP` folder:

```bash
python "Assignment 2/code.py"
python "Assignment 3/code.py"
python "Assignment 4/code.py"
python "Assignment 5/code.py"
python "Assignment 8/code.py"
python "Assignment 9/code.py"
python "Assignment 10/code.py"
```

### Notebook-based assignments

Open and run in Jupyter/VS Code Notebook:

- `Assignment 6/code.ipynb`
- `Assignment7/code.ipynb`

## Outputs

- Most assignments generate artifacts such as CSV files, plots, and model files.
- For recent assignments, outputs are saved under each assignment's local `outputs/` folder.
- Earlier assignments may save some files directly inside their assignment folder based on the original script behavior.

## Notes

- NLTK datasets are downloaded by scripts as required (first run may take longer).
- Some tasks may require internet access (for example, translation APIs or model downloads).
- Keep assignment folder names unchanged to preserve consistency with existing structure.

## Maintainer

Prepared as part of the NLP coursework assignment repository.