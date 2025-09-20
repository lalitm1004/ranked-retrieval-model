from pathlib import Path
from collections import defaultdict
import os
import spacy
import json
from typing import List, Tuple
from collections import Counter
from pydantic import BaseModel

from spacy.lang.en import English

nlp = spacy.load("en_core_web_sm")


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
                f.write(json.dumps(doc.model_dump_json()) + "\n")

    def __doc_process(self, doc_id: int, file_path: Path):
        text = file_path.read_text(encoding="utf-8")
        doc = nlp(text)

        tokens: List[str] = [
            token.lemma_
            for token in doc
            # if not any([token.is_space, token.is_stop, token.is_punct])
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
    from pathlib import Path

    # Paths
    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)  # make sure folder exists

    # Initialize preprocessing
    factory = PreprocessingFactory(docs_path=raw_path, proc_docs_path=processed_path)

    # Quick check: print first few processed documents
    print("Processed Documents:")
    for doc in factory.docs[:3]:
        print(
            f"Doc ID: {doc.doc_id}, File: {doc.file_path.name}, Tokens: {doc.tokens[:10]}"
        )

    # Quick check: print first few postings
    print("\nPosting List:")
    for posting in factory.posting_list[:5]:
        print(
            f"Token ID: {posting.token_id}, Token: {posting.token}, Postings: {posting.postings}"
        )
