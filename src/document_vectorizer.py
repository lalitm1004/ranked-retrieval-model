import math
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List

from preprocessing import ProcessedDocument, Posting


class DocumentTFVector(BaseModel):
    doc_id: int
    tf_vector: List[float]


class DocumentTFVectorizer:
    def __init__(
        self,
        processed_documents: List[ProcessedDocument],
        posting_list: Dict[str, Posting],
    ) -> None:
        self.vectors: List[DocumentTFVector] = []

        for doc in processed_documents:
            tf_vector = [0.0] * len(posting_list)

            for token in doc.tokens:
                if token not in posting_list.keys():
                    continue

                token_index = posting_list[token].token_id

                t_f = 0.0
                for doc_id, freq in posting_list[token].postings:
                    if doc_id == doc.doc_id:
                        t_f = freq
                        break

                if t_f > 0.0:
                    tf_vector[token_index] = 1 + math.log10(t_f)

            norm = math.sqrt(sum(val * val for val in tf_vector))
            if norm > 0:
                tf_vector = [val / norm for val in tf_vector]

            self.vectors.append(
                DocumentTFVector(doc_id=doc.doc_id, tf_vector=tf_vector)
            )

    def save_vectors(self, output_path: Path) -> None:
        with open(output_path, "w") as f:
            for v in self.vectors:
                f.write(v.model_dump_json(indent=4))


if __name__ == "__main__":
    from preprocessing import PreprocessingFactory

    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    factory = PreprocessingFactory(docs_path=raw_path, proc_docs_path=processed_path)

    vectorizer = DocumentTFVectorizer(factory.docs, factory.posting_list)
    vectorizer.save_vectors(Path("./data/processed/vectors.jsonl"))
