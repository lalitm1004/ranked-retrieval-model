import streamlit as st
from pathlib import Path

from document_vectorizer import DocumentTFVectorizer
from preprocessing import PreprocessingFactory
from query_vectorizer import QueryVectorizer, get_vocab_idf
from search import Search

# Apply dark theme custom styling
def apply_custom_css():
    st.markdown("""
    <style>
        /* Force dark theme throughout the app */
        .stApp {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }

        /* Containers and other background elements */
        div.block-container, div.css-1y4p8pa, div.css-1wrcr25, section.main {
            background-color: #0e1117 !important;
        }

        /* Preview box for search results */
        .preview-box {
            background-color: #1e1e1e !important;
            color: #fafafa !important;
            padding: 1rem;
            border-radius: 4px;
            font-family: monospace;
            margin-top: 0.5rem;
            border: 1px solid #3b3b3b;
        }

        /* Document content view */
        .document-content {
            font-family: monospace;
            background-color: #1e1e1e !important;
            color: #fafafa !important;
            padding: 1rem;
            border-radius: 4px;
            white-space: pre-wrap;
            border: 1px solid #3b3b3b;
        }

        /* Search results container */
        .search-results {
            margin-top: 2rem;
        }

        /* Result item separator */
        .result-item {
            padding-bottom: 1rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid #3b3b3b;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }

        /* Paragraphs */
        p {
            color: #fafafa !important;
        }

        /* Text input fields */
        .stTextInput > div > div > input {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border: 1px solid #3b3b3b !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #2c2c2c !important;
            color: #ffffff !important;
            border: 1px solid #3b3b3b !important;
        }

        /* Button hover effects */
        .stButton > button:hover {
            background-color: #3c3c3c !important;
            border: 1px solid #4c4c4c !important;
        }

        /* Spinner and other elements */
        .stSpinner > div > div, div.css-1offfwp {
            border-color: #4c4c4c !important;
        }

        /* Warning messages */
        div.element-container div.stAlert {
            background-color: #1e1e1e !important;
            color: #fafafa !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize the search engine components
@st.cache_resource
def initialize_search_engine():
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

    return preprocessing_factory, document_vectorizer, idf_values, query_vectorizer, doc_id_to_path

# Function to perform search
def perform_search(query, search_components):
    preprocessing_factory, document_vectorizer, idf_values, query_vectorizer, doc_id_to_path = search_components

    query_vector, query_tokens = query_vectorizer.vectorize_query(query)

    result = Search(
        document_vectorizer.vectors,
        query_vector,
        query_tokens,
        preprocessing_factory.posting_list,
        preprocessing_factory.soundex_posting_list,
        doc_id_to_path,
    ).search(idf_values)

    return result

# Function to read document content
def read_document(file_path):
    return file_path.read_text(encoding="utf-8")

# Function to create document preview (first 200 characters)
def get_document_preview(content):
    if len(content) <= 200:
        return content
    return content[:200] + "..."

# Search page
def search_page():
    # Title
    st.title("Ranked Retrieval")

    # Search input
    search_query = st.text_input("", placeholder="Enter your search query...", key="search_input")

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []

    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0

    # Perform search when button is clicked
    if search_button and search_query:
        search_components = initialize_search_engine()
        with st.spinner("Searching..."):
            results = perform_search(search_query, search_components)

        st.session_state.search_results = results
        st.session_state.search_query = search_query
        st.session_state.page_number = 0

    # Display results
    if 'search_results' in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results

        st.markdown(f"### Found {len(results)} results")

        # Pagination setup
        results_per_page = 5
        total_pages = (len(results) + results_per_page - 1) // results_per_page

        # Calculate page slice
        start_idx = st.session_state.page_number * results_per_page
        end_idx = min(start_idx + results_per_page, len(results))
        page_results = results[start_idx:end_idx]

        # Display results
        for i, file_path in enumerate(page_results):
            result_idx = start_idx + i

            # Read document content
            content = read_document(file_path)
            preview = get_document_preview(content)

            st.markdown("---")
            st.subheader(file_path.name)
            st.write(f"**Path:** {file_path}")
            st.write("**Preview:**")
            st.markdown(f'<div class="preview-box">{preview}</div>', unsafe_allow_html=True)

            # View document button
            if st.button("View Document", key=f"view_{result_idx}"):
                st.session_state.current_document = str(file_path)
                st.session_state.current_page = "document_view"
                st.rerun()

        st.markdown("---")

        # Pagination controls
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                if st.session_state.page_number > 0:
                    if st.button("Previous Page", key="prev_page"):
                        st.session_state.page_number -= 1
                        st.rerun()

            with col2:
                st.write(f"Page {st.session_state.page_number + 1} of {total_pages}")

            with col3:
                if st.session_state.page_number < total_pages - 1:
                    if st.button("Next Page", key="next_page"):
                        st.session_state.page_number += 1
                        st.rerun()

    # Show message if no results found
    elif 'search_query' in st.session_state and not st.session_state.search_results:
        st.warning("No results found. Try a different search query.")

# Document view page
def document_view_page():
    if 'current_document' in st.session_state:
        file_path = Path(st.session_state.current_document)

        # Back button at the top
        if st.button("← Back to Results"):
            st.session_state.current_page = "search"
            st.rerun()

        # Document title and metadata
        st.title(file_path.name)
        st.write(f"**Path:** {file_path}")

        # Read and display document
        try:
            content = read_document(file_path)
            st.markdown(f'<div class="document-content">{content}</div>', unsafe_allow_html=True)
            #st.text(content)
            #st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error reading document: {e}")

# Main app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Ranked Retrieval Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Apply custom CSS
    apply_custom_css()

    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "search"

    # Page routing
    if st.session_state.current_page == "search":
        search_page()
    elif st.session_state.current_page == "document_view":
        document_view_page()

if __name__ == "__main__":
    main()
