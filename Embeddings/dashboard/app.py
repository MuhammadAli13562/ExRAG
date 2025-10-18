"""
Vector Database Query Dashboard

A beautiful Streamlit interface for querying and visualizing ChromaDB collections.
"""
import sys
from pathlib import Path

# Add parent directory to path (for both direct run and streamlit run)
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st

try:
    from indexer import VectorIndexer
    from config import EMBEDDING_MODEL
    from embedder import Embedder
    from node_loader import get_cached_node_map, get_node_by_id
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error(f"Current working directory: {Path.cwd()}")
    st.error(f"Python path: {sys.path}")
    st.error("Please run the dashboard using: ./run_dashboard.sh")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Vector Query Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, readable styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Clean result cards */
    .stContainer {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 600;
    }
    
    /* Expander headers - clean and readable */
    .streamlit-expanderHeader {
        background-color: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        color: #1f2937 !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e5e7eb !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        border: 1px solid #e5e7eb;
        border-top: none;
        border-radius: 0 0 6px 6px;
        padding: 1rem;
        background-color: #fafafa;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1f2937;
    }
    
    /* Caption text */
    .st-emotion-cache-16idsys p {
        color: #6b7280 !important;
        font-size: 0.85rem;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Text areas */
    textarea {
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_indexer():
    """Get cached indexer instance."""
    return VectorIndexer()

@st.cache_resource
def get_embedder():
    """Get cached embedder instance for query embedding."""
    return Embedder()

@st.cache_data(ttl=3600)  # Cache embeddings for 1 hour
def generate_query_embedding(_embedder, query_text: str):
    """Generate and cache query embedding."""
    return _embedder.embed_single(query_text)


def display_result_card(rank, result_data, full_node=None, index_field="text"):
    """Display a single result as a clean expandable card."""
    doc = result_data['document']  # This is the indexed field content
    metadata = result_data['metadata']
    distance = result_data['distance']
    similarity = 1 - distance
    
    # Use full node data if available
    if full_node:
        display_title = full_node.get('title', metadata.get('title', 'Untitled'))
        full_text = full_node.get('text', '')
        summary = full_node.get('summary', '')
        prefix_summary = full_node.get('prefix_summary', '')
        line_num = full_node.get('line_num', metadata.get('line_num', 'N/A'))
    else:
        display_title = metadata.get('title', 'Untitled')
        full_text = doc if index_field == 'text' else ''
        summary = metadata.get('summary', '')
        prefix_summary = ''
        line_num = metadata.get('line_num', 'N/A')
    
    # Create unique key for this result
    card_key = f"result_{rank}_{metadata.get('node_id', rank)}"
    
    # Result card container with cleaner styling
    with st.container():
        # Header with rank, title, and similarity
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### 🎯 Rank {rank}")
            st.markdown(f"**{display_title}**")
        with col2:
            similarity_color = "#10b981" if similarity > 0.8 else "#f59e0b" if similarity > 0.6 else "#ef4444"
            st.markdown(
                f"<div style='text-align: right; padding-top: 0.5rem;'>"
                f"<span style='background: {similarity_color}; color: white; padding: 0.4rem 0.8rem; "
                f"border-radius: 20px; font-weight: 600; font-size: 0.95rem;'>Match: {similarity:.1%}</span></div>",
                unsafe_allow_html=True
            )
        
        # Compact metadata row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.caption(f"📚 Ch. {metadata.get('chapter', 'N/A')}")
        with col2:
            st.caption(f"📄 Pg. {metadata.get('page_num', 'N/A')}")
        with col3:
            st.caption(f"🔖 ID: {metadata.get('node_id', 'N/A')}")
        with col4:
            st.caption(f"📍 Line: {line_num}")
        with col5:
            anchor = metadata.get('anchor_type', '')
            if anchor:
                st.caption(f"🎯 {anchor}")
        
        st.markdown("---")
        
        # Expandable fields
        if summary and summary.strip():
            with st.expander("📝 **Summary**", expanded=False):
                st.markdown(f"""
                <div style='background: #f9fafb; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid #3b82f6; color: #1f2937; line-height: 1.6;'>
                    {summary}
                </div>
                """, unsafe_allow_html=True)
        
        if prefix_summary and prefix_summary.strip():
            with st.expander("📋 **Prefix Summary**", expanded=False):
                st.markdown(f"""
                <div style='background: #f9fafb; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid #8b5cf6; color: #1f2937; line-height: 1.6;'>
                    {prefix_summary}
                </div>
                """, unsafe_allow_html=True)
        
        if full_text and full_text.strip():
            with st.expander("📄 **Full Text**", expanded=False):
                st.markdown(f"""
                <div style='background: #f9fafb; padding: 1rem; border-radius: 8px; 
                            border-left: 3px solid #10b981; color: #1f2937; line-height: 1.6; 
                            font-family: "SF Mono", monospace; font-size: 0.9rem; white-space: pre-wrap;'>
                    {full_text}
                </div>
                """, unsafe_allow_html=True)
        
        # Indexed field indicator
        with st.expander("ℹ️ **Metadata & Details**", expanded=False):
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.text(f"Indexed Field: {index_field}")
                st.text(f"Section: {metadata.get('section', 'N/A')}")
                st.text(f"Heading Level: {metadata.get('heading_level', 'N/A')}")
            with meta_col2:
                st.text(f"Token Count: {metadata.get('token_count', 'N/A')}")
                st.text(f"All Pages: {metadata.get('all_pages', 'N/A')}")
                st.text(f"Distance: {distance:.4f}")
        
        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 2px solid #e5e7eb;'>", unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown("# 🔍 Vector Database Query Dashboard")
    st.markdown("**Explore and visualize your ChromaDB collections interactively**")
    
    # Initialize indexer and embedder
    try:
        indexer = get_indexer()
        embedder = get_embedder()
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Collection selector
        collections = indexer.list_collections()
        
        if not collections:
            st.warning("⚠️ No collections found. Please index a JSON file first.")
            st.code("python run_indexer.py your_file.json", language="bash")
            st.stop()
        
        selected_collection = st.selectbox(
            "📊 Select Collection",
            collections,
            help="Choose the ChromaDB collection to query"
        )
        
        # Get collection info
        if selected_collection:
            collection_info = indexer.get_collection_info(selected_collection)
            collection_metadata = collection_info.get('metadata', {})
            index_field = collection_metadata.get('index_field', 'text')
            
            st.markdown("### 📈 Collection Stats")
            st.metric("Total Vectors", collection_info.get('count', 0))
            st.metric("Embedding Model", EMBEDDING_MODEL)
            st.metric("🎯 Indexed Field", index_field.title())
            
            # Show metadata
            with st.expander("ℹ️ Collection Details"):
                st.json(collection_metadata)
        
        st.markdown("---")
        
        # Query parameters
        st.markdown("### 🎛️ Query Settings")
        
        top_k = st.slider(
            "📊 Top K Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of results to retrieve"
        )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            filter_by_chapter = st.text_input("Filter by Chapter", placeholder="e.g., 7")
            filter_by_anchor = st.selectbox(
                "Filter by Anchor Type",
                ["None", "Experiment", "Discussion", "Example", "Exercise", "DSE exam", "DSE goal"]
            )
            filter_by_page = st.number_input("Filter by Page", min_value=0, value=0, step=1)
    
    # Main query interface
    st.markdown("## 💬 Query Interface")
    
    query_text = st.text_input(
        "🔎 Enter your question:",
        placeholder="e.g., How does electromagnetic induction work?",
        help="Type your natural language question here"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()
    
    # Perform search
    if search_button and query_text and selected_collection:
        import time
        start_time = time.time()
        
        with st.spinner("🔍 Searching vector database..."):
            try:
                # Get collection and its metadata
                collection = indexer.client.get_collection(selected_collection)
                collection_info = indexer.get_collection_info(selected_collection)
                search_metadata = collection_info.get('metadata', {})
                
                # Build filter conditions
                where_filter = {}
                if filter_by_chapter:
                    where_filter["chapter"] = filter_by_chapter
                if filter_by_anchor and filter_by_anchor != "None":
                    where_filter["anchor_type"] = filter_by_anchor
                if filter_by_page > 0:
                    where_filter["page_num"] = filter_by_page
                
                # Generate query embedding (fixes dimension mismatch & improves speed)
                embed_start = time.time()
                query_embedding = generate_query_embedding(embedder, query_text)
                embed_time = time.time() - embed_start
                
                # Query using pre-computed embedding (much faster!)
                query_start = time.time()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                    where=where_filter if where_filter else None
                )
                query_time = time.time() - query_start
                total_time = time.time() - start_time
                
                # Load node map from JSON source if available
                node_map = None
                json_source = search_metadata.get('json_source')
                if json_source:
                    try:
                        from pathlib import Path
                        json_path = Path(json_source)
                        
                        # Try multiple path resolutions
                        possible_paths = [
                            json_path,  # As-is
                            Path.cwd() / json_path,  # Relative to current dir
                            Path.cwd().parent / json_path,  # Relative to parent dir
                            Path(__file__).parent.parent / json_path,  # Relative to C-Embeddings parent
                        ]
                        
                        found_path = None
                        for path in possible_paths:
                            if path.exists():
                                found_path = path
                                break
                        
                        if found_path:
                            node_map = get_cached_node_map(str(found_path))
                            st.success(f"✅ Loaded {len(node_map)} nodes from: {found_path.name}")
                        else:
                            st.warning(f"⚠️ JSON source not found. Tried: {json_source}")
                            st.caption(f"Searched in: {', '.join(str(p.parent) for p in possible_paths[:3])}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load source JSON for full node data: {e}")
                        st.exception(e)
                else:
                    st.info("ℹ️ No JSON source in collection metadata. Re-index to enable full node display.")
                
                # Display results
                if results and results['documents'][0]:
                    st.markdown("---")
                    st.markdown(f"## 📊 Results ({len(results['documents'][0])} found)")
                    
                    # Summary stats with performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_similarity = sum(1 - d for d in results['distances'][0]) / len(results['distances'][0])
                        st.metric("Average Similarity", f"{avg_similarity:.1%}")
                    with col2:
                        best_similarity = 1 - min(results['distances'][0])
                        st.metric("Best Match", f"{best_similarity:.1%}")
                    with col3:
                        st.metric("Results Returned", len(results['documents'][0]))
                    with col4:
                        st.metric("⚡ Total Time", f"{total_time:.2f}s")
                    
                    # Performance breakdown
                    with st.expander("⏱️ Performance Details"):
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        with perf_col1:
                            st.metric("Embedding Generation", f"{embed_time:.3f}s")
                        with perf_col2:
                            st.metric("Vector Search", f"{query_time:.3f}s")
                        with perf_col3:
                            overhead = total_time - embed_time - query_time
                            st.metric("Overhead", f"{overhead:.3f}s")
                    
                    st.markdown("---")
                    
                    # Display each result
                    for i, (doc, metadata, distance) in enumerate(
                        zip(results['documents'][0], results['metadatas'][0], results['distances'][0]),
                        1
                    ):
                        result_data = {
                            'document': doc,
                            'metadata': metadata,
                            'distance': distance
                        }
                        
                        # Get full node if node_map available
                        full_node = None
                        if node_map and metadata.get('node_id'):
                            full_node = get_node_by_id(node_map, metadata['node_id'])
                            if not full_node:
                                st.warning(f"⚠️ Node {metadata['node_id']} not found in source JSON")
                        
                        # Get index field from metadata
                        index_field = search_metadata.get('index_field', 'text')
                        
                        # Debug info in expander
                        if i == 1:  # Only show for first result
                            with st.expander("🔍 Debug Info (First Result)", expanded=False):
                                st.text(f"Node Map Loaded: {node_map is not None}")
                                st.text(f"Node ID: {metadata.get('node_id')}")
                                st.text(f"Full Node Retrieved: {full_node is not None}")
                                if full_node:
                                    st.text(f"Full Node has 'text': {'text' in full_node}")
                                    st.text(f"Text length: {len(full_node.get('text', ''))}")
                                    st.json({k: f"{str(v)[:50]}..." if len(str(v)) > 50 else v 
                                            for k, v in full_node.items() if k != 'text'})
                        
                        display_result_card(i, result_data, full_node=full_node, index_field=index_field)
                
                else:
                    st.warning("⚠️ No results found. Try adjusting your query or filters.")
                    
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                st.exception(e)
    
    elif search_button and not query_text:
        st.warning("⚠️ Please enter a question to search.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>"
        "Built with ❤️ using Streamlit & ChromaDB | "
        f"Model: {EMBEDDING_MODEL}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

