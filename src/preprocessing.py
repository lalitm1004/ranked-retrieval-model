from pathlib import Path
from collections import defaultdict, Counter
import os
import json
from typing import List, Tuple
from pydantic import BaseModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punct_set = set(string.punctuation)


class ProcessedDocument(BaseModel):
    file_path: Path
    doc_id: int
    tokens: List[str]


class RawPosting(BaseModel):
    token: str
    doc_id: int
    t_f: int


class Posting(BaseModel):
    token_id: int
    token: str
    postings: List[Tuple[int, int]]  # tuple(doc_id, t_f)


class PreprocessingFactory:
    def __init__(self, docs_path: Path, proc_docs_path: Path):
        self.doc_dir: Path = docs_path
        self.proc_docs_path: Path = proc_docs_path

        self.docs: List[ProcessedDocument] = []
        self.postings: List[RawPosting] = []

        self.__dir_process()
        self.posting_list = self.__posting_list()

    def __dir_process(self):
        for index, file_name in enumerate(os.listdir(self.doc_dir)):
            file_path: Path = self.doc_dir / file_name

            if file_path.is_file():
                self.__doc_process(doc_id=index, file_path=file_path)

        output_file = self.proc_docs_path / "docs.jsonl"
        with open(Path(output_file), "w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(doc.model_dump_json(indent=4))

    def __doc_process(self, doc_id: int, file_path: Path):
        text = file_path.read_text(encoding="utf-8")

        raw_tokens = word_tokenize(text)

        tokens: List[str] = [
            lemmatizer.lemmatize(token.lower())
            for token in raw_tokens
            if token.lower() not in stop_words
            and token not in punct_set
            and token.strip() != ""
        ]

        processed_doc = ProcessedDocument(
            file_path=file_path, doc_id=doc_id, tokens=tokens
        )
        self.docs.append(processed_doc)

        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            posting = RawPosting(token=token, doc_id=doc_id, t_f=count)
            self.postings.append(posting)

    def __posting_list(self):
        grouped_postings = defaultdict(list)
        for raw_posting in self.postings:
            grouped_postings[raw_posting.token].append(
                (raw_posting.doc_id, raw_posting.t_f)
            )

        posting_list = []
        for idx, (token, postings) in enumerate(grouped_postings.items()):
            postings = sorted(postings, key=lambda x: x[1], reverse=True)
            posting_list.append(Posting(token_id=idx, token=token, postings=postings))

        return posting_list


if __name__ == "__main__":
    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = PreprocessingFactory(docs_path=raw_path, proc_docs_path=processed_path)

    print("Processed Documents:")
    for doc in factory.docs:
        print(
            f"Doc ID: {doc.doc_id}, File: {doc.file_path.name}, Tokens: {doc.tokens[:10]}"
        )

    print("\nPosting List:")
    for posting in factory.posting_list:
        print(
            f"Token ID: {posting.token_id}, Token: {posting.token}, Postings: {posting.postings}"
        )
