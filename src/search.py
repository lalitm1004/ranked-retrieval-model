from typing import Dict, List, Set
from config import Config
import numpy as np

from preprocessing import Posting, SoundexPosting
from document_vectorizer import DocumentTFVector
from query_vectorizer import QueryToken


def get_cutoff_idf(idf_values: Dict[str, float]) -> float:
    idf_scores = list(idf_values.values())

    if not idf_scores:
        return 0.0

    idf_scores.sort(reverse=True)

    percentile = Config.top_percentile / 100.0
    cutoff_index = int(len(idf_scores) * percentile)

    if cutoff_index < 1:
        cutoff_index = 1
    if cutoff_index > len(idf_scores):
        cutoff_index = len(idf_scores)

    return idf_scores[cutoff_index - 1]


class Search:
    def __init__(
        self,
        document_vectors: List[DocumentTFVector],
        query_vector: List[float],
        query_tokens: List[QueryToken],
        posting_list: Dict[str, Posting],
        soundex_posting_list: Dict[str, SoundexPosting],
    ) -> None:
        # Convert all document vectors to NumPy arrays once
        self.document_vectors = {
            dv.doc_id: np.array(dv.tf_vector, dtype=float) for dv in document_vectors
        }
        # Convert query vector to NumPy array once
        self.query_vector = np.array(query_vector, dtype=float)

        self.query_tokens = query_tokens
        self.posting_list = posting_list
        self.soundex_posting_list = soundex_posting_list

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, idf_values: Dict[str, float]) -> List[int]:
        # 1. Get cutoff idf
        idf_scores = list(idf_values.values())
        if not idf_scores:
            return []
        idf_scores.sort(reverse=True)
        percentile = Config.top_percentile / 100.0

        cutoff_index = int(len(idf_scores) * percentile)
        cutoff_index = max(1, min(cutoff_index, len(idf_scores)))
        cutoff_idf = idf_scores[cutoff_index - 1]

        # 2. Sort query tokens descending by idf_score
        tokens_sorted = sorted(
            self.query_tokens, key=lambda x: x.idf_score, reverse=True
        )

        # 3. Gather doc_ids from top query tokens above cutoff
        candidate_doc_ids: Set[int] = set()
        for qt in tokens_sorted:
            if qt.idf_score < cutoff_idf:
                continue
            if qt.is_soundex:
                if qt.token not in self.soundex_posting_list:
                    continue
                soundex_post = self.soundex_posting_list[qt.token]
                for doc_id, _ in soundex_post.postings:
                    candidate_doc_ids.add(doc_id)
            else:
                if qt.token not in self.posting_list:
                    continue
                post = self.posting_list[qt.token]
                for doc_id, _ in post.postings:
                    candidate_doc_ids.add(doc_id)
            if len(candidate_doc_ids) >= Config.max_fetch_count:
                break

        # 4. Compute cosine similarity for each candidate doc
        scored_docs = []
        for doc_id in candidate_doc_ids:
            if doc_id not in self.document_vectors:
                continue
            doc_vec = self.document_vectors[doc_id]
            sim = self._cosine_similarity(self.query_vector, doc_vec)
            scored_docs.append((doc_id, sim))

        # 5. Sort by similarity descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, _ in scored_docs]


import matplotlib.pyplot as plt
import numpy as np


def plot_idf_distribution(idf_dict: Dict[str, float], top_n: int = 50) -> None:
    # Sort tokens by IDF descending
    sorted_items = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)

    # Optionally take only top_n for readability
    tokens, idfs = zip(*sorted_items[:top_n])

    # Create positions for the x-axis
    x_positions = np.arange(len(tokens))

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(x_positions, idfs, color="steelblue")

    # Label axes and title
    plt.xlabel("Tokens")
    plt.ylabel("IDF Score")
    plt.title(f"Top {top_n} Tokens by IDF Score")

    # Rotate x-axis labels for readability
    plt.xticks(x_positions, tokens, rotation=90)

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Show the plot
    plt.show()


# Usage after you compute idf_dict:


if __name__ == "__main__":
    from preprocessing import PreprocessingFactory
    from pathlib import Path
    from query_vectorizer import QueryVectorizer, get_vocab_idf
    from document_vectorizer import DocumentTFVectorizer

    query = "As of August 10, 2015, Google is a subsidiary of the Alphabet Inc."

    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = PreprocessingFactory(docs_path=raw_path, proc_docs_path=processed_path)

    dv = DocumentTFVectorizer(factory.docs, factory.posting_list)

    idf_dict = get_vocab_idf(factory.posting_list)
    print(list(idf_dict.values()))

    # plot_idf_distribution(idf_dict, top_n=50)

    qv = QueryVectorizer(
        factory.posting_list,
        factory.soundex_posting_list,
        idf_dict,
        idw_map=factory.idw_map,
    )

    a = qv.vectorize_query(query)

    s = Search(
        dv.vectors, a[0], a[1], factory.posting_list, factory.soundex_posting_list
    )
    res = s.search(idf_dict)
    # print(res)
