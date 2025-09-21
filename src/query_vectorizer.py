import math
from pydantic import BaseModel
from typing import Dict, List
from collections import Counter

from preprocessing import Posting, token_processing_helper


class QueryVector(BaseModel):
    query: str
    tfidf_vector: List[float]


class QueryVectorizer:
    def __init__(self, posting_list: Dict[str, Posting]) -> None:
        self.posting_list = posting_list
        self.total_docs = max(
            max((doc_id for doc_id, _ in posting.postings), default=0)
            for posting in posting_list.values()
        )

    def vectorize_query(self, query: str) -> List[float]:
        tokens = token_processing_helper(query)

        tf_counter = Counter(tokens)

        vector = [0.0] * len(self.posting_list)

        for token, tf in tf_counter.items():
            if token not in self.posting_list:
                continue

            posting = self.posting_list[token]
            index = posting.token_id

            term_freq = 1 + math.log10(tf)
            df = len(posting.postings)

            idf = math.log10(self.total_docs / (1 + df))

            vector[index] = term_freq * idf

        return vector


if __name__ == "__main__":
    from preprocessing import PreprocessingFactory
    from pathlib import Path

    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = PreprocessingFactory(docs_path=raw_path, proc_docs_path=processed_path)

    vectorizer = QueryVectorizer(factory.posting_list)
    a = vectorizer.vectorize_query("Adobe HP Dell Google Internet")
    print(a)
