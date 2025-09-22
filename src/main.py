from pathlib import Path

from document_vectorizer import DocumentTFVectorizer
from preprocessing import PreprocessingFactory
from query_vectorizer import QueryVectorizer, get_vocab_idf
from search import Search


def main():
    raw_path = Path("./data/raw")
    processed_path = Path("./data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    preprocessing_factory = PreprocessingFactory(
        docs_path=raw_path, proc_docs_path=processed_path
    )

    document_vectorizer = DocumentTFVectorizer(
        preprocessing_factory.docs, preprocessing_factory.posting_list
    )

    idf_values = get_vocab_idf(preprocessing_factory.posting_list)

    query_vectorizer = QueryVectorizer(
        preprocessing_factory.posting_list,
        preprocessing_factory.soundex_posting_list,
        idf_values,
        preprocessing_factory.idw_map,
    )
    doc_id_to_path = {doc.doc_id: doc.file_path for doc in preprocessing_factory.docs}

    while True:
        print("Enter query > ", end="")
        query = input().strip()
        query_vector, query_tokens = query_vectorizer.vectorize_query(query)

        result = Search(
            document_vectorizer.vectors,
            query_vector,
            query_tokens,
            preprocessing_factory.posting_list,
            preprocessing_factory.soundex_posting_list,
            doc_id_to_path,
        ).search(idf_values)

        for file_path, cosine_similarity in result:
            print(f"PATH: {file_path}, SIMILARITY: {cosine_similarity}")

        print()


if __name__ == "__main__":
    main()
