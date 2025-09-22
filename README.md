# Information Retrieval Assignment 1 - Ranked Retrieval System
**Course**: CSD358 - Information Retrieval Lab  
**Assignment**: Vector Space Model Implementation  
**Due Date**: September 22, 2025, 10:00 PM
**DEPLOYMENT LINK**: https://ranked-retrieval-model-ir.streamlit.app/
**GITHUB LINK**: https://github.com/lalitm1004/ranked-retrieval-model

## Project Overview

This project implements a ranked retrieval system using the Vector Space Model (VSM) with TF-IDF weighting scheme. The system processes a document corpus, builds inverted indices, and provides ranked search results based on cosine similarity between query and document vectors.

### Key Features

- **Vector Space Model**: Implementation of lnc.ltc ranking scheme
- **TF-IDF Weighting**: Log term frequency with IDF for queries, log TF with cosine normalization for documents
- **Soundex Algorithm**: Spelling matching for improved query handling
- **Web Interface**: Streamlit-based GUI for easy interaction
- **Document Preprocessing**: Lemmatization, stopword removal, and POS tagging
- **Ranked Results**: Returns up to 10 most relevant documents ordered by cosine similarity

## System Architecture

### Core Components

1. **preprocessing.py**: Document processing and inverted index creation
2. **document_vectorizer.py**: TF vector computation for documents  
3. **query_vectorizer.py**: Query vector computation with TF-IDF weighting
4. **search.py**: Cosine similarity calculation and ranking
5. **app.py**: Streamlit web interface
6. **main.py**: Command-line interface
7. **config.py**: System configuration parameters

### Mathematical Implementation

#### TF-IDF Calculation
- **Term Frequency**: `tf = 1 + log(raw_tf)` (for tf > 0)
- **Inverse Document Frequency**: `idf = log(N/df)`
- **Document Vectors**: Log TF with cosine normalization (lnc)
- **Query Vectors**: Log TF × IDF with cosine normalization (ltc)

#### Cosine Similarity
```
similarity(q,d) = (q · d) / (|q| × |d|)
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- Required packages listed in requirements (see dependencies below)

### Dependencies
```bash
pip install -r requirements.txt
```

### NLTK Data Download
The system automatically downloads required NLTK data:
- punkt (tokenization)
- stopwords
- wordnet (lemmatization)  
- averaged_perceptron_tagger_eng (POS tagging)

## Directory Structure

```
project/
├── src/
│   ├── app.py              # Streamlit web interface
│   ├── main.py             # Command-line interface
│   ├── preprocessing.py     # Document processing
│   ├── document_vectorizer.py # Document TF vectors
│   ├── query_vectorizer.py  # Query TF-IDF vectors
│   ├── search.py           # Search and ranking
│   └── config.py           # Configuration
├── data/
│   ├── raw/               # Input document corpus
│   └── processed/         # Generated indices and vectors
└── README.md
```

## Running the Application

### Method 1: Streamlit Web Interface (Recommended)

**Run Streamlit Application**:
```bash
streamlit run src/app.py
```

### Method 2: Command Line Interface

**Run CLI Version**:
```bash
python src/main.py
```

## System Configuration

The system uses configurable parameters in `config.py`:

```python
@dataclass
class Config:
    top_percentile: float = 80.0    # IDF cutoff for candidate selection
    max_fetch_pool: int = 20        # Maximum candidate documents
    max_fetch_count: int = 10       # Maximum results returned
```

## Implementation Details

### Text Preprocessing Pipeline

1. **Tokenization**: NLTK word tokenization with hyphen handling
2. **POS Tagging**: Part-of-speech tagging for accurate lemmatization
3. **Filtering**: Remove stopwords, punctuation, and empty tokens
4. **Lemmatization**: WordNet lemmatizer with POS-aware processing
5. **Normalization**: Convert to lowercase

### Inverted Index Structure

The system maintains two types of inverted indices:

1. **Standard Posting Lists**:
   ```python
   {
       "token": Posting(
           token_id: int,
           postings: [(doc_id, term_freq), ...]
       )
   }
   ```

2. **Soundex Posting Lists**:
   ```python
   {
       "soundex_code": SoundexPosting(
           token_ids: [int, ...],
           postings: [(doc_id, combined_freq), ...]
       )
   }
   ```

### Search Algorithm

1. **Query Processing**: Same preprocessing pipeline as documents
2. **Vector Construction**: TF-IDF weighted query vector
3. **Candidate Selection**: Use high-IDF terms for efficiency
4. **Similarity Computation**: Cosine similarity with all candidates
5. **Ranking**: Sort by similarity score (descending)
6. **Result Limiting**: Return top 10 documents

### Soundex Integration

- Handles spelling variations and typos
- Maps phonetically similar words to same code
- Distributes query weights across similar terms
- Proportional weighting based on original term IDF values

## Technical Specifications

### SMART Notation Compliance
- **Documents**: lnc (log tf, no idf, cosine normalization)
- **Queries**: ltn (log tf, idf)

### Logarithm Base
- All logarithmic calculations use base 10 as specified

### Result Ordering
- Primary sort: Cosine similarity (descending)
- Secondary sort: Document ID (ascending) for ties

## Assignment Compliance

This implementation fully satisfies the assignment requirements:

✅ **Vector Space Model**: Complete VSM implementation  
✅ **TF-IDF Weighting**: Proper lnc.ltn scheme  
✅ **Cosine Similarity**: Accurate similarity computation  
✅ **Document Normalization**: Length normalization implemented  
✅ **Free Text Queries**: Natural language query support  
✅ **Top 10 Results**: Relevance-ordered result limiting  
✅ **Soundex Algorithm**: Spelling matching capability  

## Conclusion

This ranked retrieval system provides a robust implementation of the Vector Space Model with modern web interface capabilities. The system efficiently processes document corpora, builds comprehensive indices, and delivers relevant search results through both web and command-line interfaces.

---

**Group Members**: 
- Jia Khot, 2310110491
- Lalit Maurya, 2310110164
- Rachit Kumar, 2310110234