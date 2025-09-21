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


nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punct_set = set(string.punctuation)

# top 5000 most common English words
common_words = set(top_n_list("en", 5000))


class ProcessedDocument(BaseModel):
    file_path: Path
    doc_id: int
    tokens: List[str]


class RawPosting(BaseModel):
    token: str
    doc_id: int
    t_f: int


class Posting(BaseModel):
    postings: List[Tuple[int, int]]  # (doc_id, term frequency)

class SoundexPosting(BaseModel):
    token_ids: List[int]
    postings: List[Tuple[int, int]]  # (doc_id, term frequency)


def pos_mapping_helper(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def token_processing_helper(raw_text: str) -> List[str]:
    raw_tokens = word_tokenize(raw_text)
    pos_tags = pos_tag(raw_tokens)

    processed_tokens: List[str] = []

    for token, tag in pos_tags:
        if (
            token.lower() not in stop_words
            and token not in punct_set
            and token.strip() != ""
        ):
            token_lemma = lemmatizer.lemmatize(
                token.lower(), pos=pos_mapping_helper(tag)
            )
            processed_tokens.append(token_lemma)

    return processed_tokens


class PreprocessingFactory:
    def __init__(self, docs_path: Path, proc_docs_path: Path) -> None:
        self.doc_dir: Path = docs_path
        self.proc_docs_path: Path = proc_docs_path

        self.docs: List[ProcessedDocument] = []
        self.postings: List[RawPosting] = []
        self.idw_map: Dict[int, str] = {}

        self.__dir_process()

        posting_list, soundex_posting_list = self.__posting_list()
        self.posting_list: Dict[str, Posting] = posting_list
        self.soundex_posting_list: Dict[str, SoundexPosting] = soundex_posting_list

    def __dir_process(self) -> None:
        for index, file_name in enumerate(os.listdir(self.doc_dir)):
            file_path: Path = self.doc_dir / file_name

            if file_path.is_file():
                self.__doc_process(doc_id=index, file_path=file_path)

        output_file = self.proc_docs_path / "docs.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(doc.model_dump_json(indent=4))

    def __doc_process(self, doc_id: int, file_path: Path) -> None:
        text = file_path.read_text(encoding="utf-8")

        tokens: List[str] = token_processing_helper(text)

        processed_doc = ProcessedDocument(
            file_path=file_path, doc_id=doc_id, tokens=tokens
        )
        self.docs.append(processed_doc)

        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            posting = RawPosting(token=token, doc_id=doc_id, t_f=count)
            self.postings.append(posting)

    def __posting_list(self) -> Tuple[Dict[str, Posting], Dict[str, SoundexPosting]]:
        grouped_postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for raw_posting in self.postings:
            grouped_postings[raw_posting.token].append(
                (raw_posting.doc_id, raw_posting.t_f)
            )

        posting_list: Dict[str, Posting] = {}
        soundex_posting_list: Dict[str, SoundexPosting] = {}

        for idx, (token, postings) in enumerate(grouped_postings.items()):
            postings_sorted = sorted(postings, key=lambda x: x[1], reverse=True)
            posting_list[token] = Posting(postings=postings_sorted)
            self.idw_map[idx] = token

            token_soundex = jellyfish.soundex(token)

            if token_soundex in soundex_posting_list:
                existing_postings = {doc_id: t_f for doc_id, t_f in soundex_posting_list[token_soundex].postings}
                soundex_posting_list[token_soundex].token_ids.append(idx)

                for doc_id, t_f in postings_sorted:
                    existing_postings[doc_id] = existing_postings.get(doc_id, 0) + t_f

                updated_postings = sorted(list(existing_postings.items()), key=lambda x: x[1], reverse=True)
                soundex_posting_list[token_soundex].postings = updated_postings
            else:
                soundex_posting_list[token_soundex] = SoundexPosting(token_ids=[idx], postings=postings_sorted)

        return posting_list, soundex_posting_list

if __name__ == "__main__":
    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = PreprocessingFactory(docs_path=raw_path, proc_docs_path=processed_path)