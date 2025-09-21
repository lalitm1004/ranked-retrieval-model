import math
import jellyfish
from collections import Counter
from pydantic import BaseModel
from typing import Dict, List, Tuple

from preprocessing import Posting, SoundexPosting, token_processing_helper


class QueryToken(BaseModel):
    token: str
    idf_score: float
    is_soundex: bool


def get_vocab_idf(posting_list: Dict[str, Posting]) -> Dict[str, float]:
    all_doc_ids = set()
    for post in posting_list.values():
        for doc_id, _ in post.postings:
            all_doc_ids.add(doc_id)

    N = len(all_doc_ids)

    idf_dict: Dict[str, float] = {}
    for token, post in posting_list.items():
        df = len(post.postings)
        idf_dict[token] = math.log10(N / df) if df > 0 else 0.0

    return idf_dict


class QueryVectorizer:
    def __init__(
        self,
        posting_list: Dict[str, Posting],
        soundex_posting_list: Dict[str, SoundexPosting],
        idf_dict: Dict[str, float],
        idw_map: Dict[int, str],
    ) -> None:
        self.posting_list = posting_list
        self.soundex_posting_list = soundex_posting_list
        self.idf_dict = idf_dict
        self.idw_map = idw_map

        all_doc_ids = set()
        for post in posting_list.values():
            for doc_id, _ in post.postings:
                all_doc_ids.add(doc_id)

        self.N = len(all_doc_ids)

    def vectorize_query(self, query: str) -> Tuple[List[float], List[QueryToken]]:
        tokens = token_processing_helper(query)

        tokens_counter = Counter(tokens)

        query_vector = [0.0] * len(self.posting_list)
        query_tokens: List[QueryToken] = []

        for token, tf in tokens_counter.items():
            if token in self.posting_list:
                posting = self.posting_list[token]
                index = posting.token_id

                idf = self.idf_dict[token]

                query_tokens.append(
                    QueryToken(token=token, idf_score=idf, is_soundex=False)
                )

                tf_weight = 1 + math.log10(tf) if tf > 0 else 0.0

                query_vector[index] += tf_weight * idf

            else:
                soundex = jellyfish.soundex(token)

                if soundex not in self.soundex_posting_list:
                    continue

                posting = self.soundex_posting_list[soundex]

                tf_weight = 1 + math.log10(tf) if tf > 0 else 0.0
                idf = math.log10(self.N / len(posting.postings))

                query_tokens.append(
                    QueryToken(token=soundex, idf_score=idf, is_soundex=True)
                )

                weight = tf_weight * idf

                token_id_idfs = []
                for token_id in posting.token_ids:
                    mapped_token = self.idw_map.get(token_id)
                    if not mapped_token:
                        continue
                    token_id_idfs.append(
                        (token_id, self.idf_dict.get(mapped_token, 0.0))
                    )

                    total_idf = sum(i[0] for i in token_id_idfs)
                    if total_idf == 0:
                        # split evenly
                        split_weight = weight / len(token_id_idfs)
                        for token_id, _ in token_id_idfs:
                            query_vector[token_id] += split_weight
                    else:
                        for token_id, token_idf in token_id_idfs:
                            proportion = token_idf / total_idf
                            query_vector[token_id] += weight * proportion

        query_tokens.sort(key=lambda x: x.idf_score, reverse=True)

        return query_vector, query_tokens


if __name__ == "__main__":
    from preprocessing import PreprocessingFactory
    from pathlib import Path

    query = "What is the role of adobe and google in the internet? Gas."

    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = PreprocessingFactory(docs_path=raw_path, proc_docs_path=processed_path)

    idf_dict = get_vocab_idf(factory.posting_list)

    qv = QueryVectorizer(
        factory.posting_list,
        factory.soundex_posting_list,
        idf_dict,
        idw_map=factory.idw_map,
    )

    a = qv.vectorize_query(query)
    # print(a)
