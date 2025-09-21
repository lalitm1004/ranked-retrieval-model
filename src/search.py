from typing import Dict, List, Set
from config import Config
import numpy as np
from pathlib import Path

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
        doc_id_to_path: Dict[int, Path],
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

        self.doc_id_to_path = doc_id_to_path

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, idf_values: Dict[str, float]) -> List[Path]:
        idf_scores = list(idf_values.values())
        if not idf_scores:
            return []

        idf_scores.sort(reverse=True)
        percentile = Config.top_percentile / 100.0
        cutoff_index = int(len(idf_scores) * percentile)
        cutoff_index = max(1, min(cutoff_index, len(idf_scores)))
        cutoff_idf = idf_scores[cutoff_index - 1]

        tokens_sorted = sorted(
            self.query_tokens, key=lambda x: x.idf_score, reverse=True
        )

        candidate_doc_ids: Set[int] = set()

        # First pass: top-IDF tokens
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
            if len(candidate_doc_ids) >= Config.max_fetch_pool:
                break

        # Fallback: lower-IDF tokens if no candidates found
        if not candidate_doc_ids:
            for qt in tokens_sorted:
                if qt.idf_score >= cutoff_idf:
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
                if len(candidate_doc_ids) >= Config.max_fetch_pool:
                    break

        # Compute cosine similarity for all candidates
        scored_docs = []
        for doc_id in candidate_doc_ids:
            if doc_id not in self.document_vectors:
                continue
            doc_vec = self.document_vectors[doc_id]
            sim = self._cosine_similarity(self.query_vector, doc_vec)
            scored_docs.append((doc_id, sim))

        # Sort by similarity descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return only top 10 paths
        result_paths: List[Path] = []
        for doc_id, _ in scored_docs[:10]:
            path = self.doc_id_to_path.get(doc_id)
            if path:
                result_paths.append(path)

        return result_paths


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

    qv = QueryVectorizer(
        factory.posting_list,
        factory.soundex_posting_list,
        idf_dict,
        idw_map=factory.idw_map,
    )

    a = qv.vectorize_query(query)

    doc_id_to_path = {doc.doc_id: doc.file_path for doc in factory.docs}

    s = Search(
        dv.vectors,
        a[0],
        a[1],
        factory.posting_list,
        factory.soundex_posting_list,
        doc_id_to_path,
    )

    for i in s.search(idf_dict):
        print(i)
