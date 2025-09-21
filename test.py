import os
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import jellyfish
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from wordfreq import top_n_list

from src.preprocessing import token_processing_helper

nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")

# a = word_tokenize("what is the role of google on the internet")
# print(a)

b = token_processing_helper("what is the role of adobe hp and google on the internet")
print(b)
