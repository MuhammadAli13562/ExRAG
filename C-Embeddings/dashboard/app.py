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

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .result-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .similarity-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .metadata-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    
    .metadata-label {
        font-weight: 600;
        opacity: 0.8;
    }
    
    .text-preview {
        background: rgba(255, 255, 255, 0.15);
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
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


def display_result_card(rank, result_data):
    """Display a single result as a beautiful card."""
    doc = result_data['document']
    metadata = result_data['metadata']
    distance = result_data['distance']
    similarity = 1 - distance
    
    # Create unique key for this result
    card_key = f"result_{rank}_{metadata.get('node_id', rank)}"
    
    # Result card container
    with st.container():
        # Header with rank and similarity
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 🎯 Rank {rank}: {metadata.get('title', 'Untitled')}")
        with col2:
            similarity_color = "#4ade80" if similarity > 0.8 else "#fbbf24" if similarity > 0.6 else "#f87171"
            st.markdown(
                f"<div style='text-align: right; font-size: 1.1rem;'>"
                f"<span style='background: {similarity_color}; color: white; padding: 0.25rem 0.75rem; "
                f"border-radius: 20px; font-weight: 600;'>✨ {similarity:.1%}</span></div>",
                unsafe_allow_html=True
            )
        
        # Metadata grid
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📚 Chapter", metadata.get('chapter', 'N/A'))
        with col2:
            st.metric("📄 Page", metadata.get('page_num', 'N/A'))
        with col3:
            st.metric("🔖 Node ID", metadata.get('node_id', 'N/A'))
        with col4:
            st.metric("📍 Line", metadata.get('line_num', 'N/A'))
        with col5:
            anchor = metadata.get('anchor_type', 'N/A')
            if anchor and anchor != 'N/A':
                st.metric("🎯 Type", anchor)
            else:
                st.metric("📊 Tokens", metadata.get('token_count', 'N/A'))
        
        # Text preview (first 300 chars)
        preview_length = 300
        preview = doc[:preview_length]
        if len(doc) > preview_length:
            preview += "..."
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin: 1rem 0;'>
            <div style='font-family: "SF Pro Display", -apple-system, sans-serif; line-height: 1.6; color: #1f2937;'>
                {preview}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandable full text
        with st.expander("📖 View Full Text"):
            st.text_area(
                "Full Content",
                value=doc,
                height=200,
                key=f"text_{card_key}",
                label_visibility="collapsed"
            )
            
            # Additional metadata
            st.markdown("**Additional Metadata:**")
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.text(f"Section: {metadata.get('section', 'N/A')}")
                st.text(f"Heading Level: {metadata.get('heading_level', 'N/A')}")
            with meta_col2:
                st.text(f"All Pages: {metadata.get('all_pages', 'N/A')}")
                st.text(f"Summary Available: {'Yes' if metadata.get('summary') else 'No'}")
        
        st.markdown("---")


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
            
            st.markdown("### 📈 Collection Stats")
            st.metric("Total Vectors", collection_info.get('count', 0))
            st.metric("Embedding Model", EMBEDDING_MODEL)
            
            # Show metadata
            with st.expander("ℹ️ Collection Details"):
                st.json(collection_info.get('metadata', {}))
        
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
                # Get collection
                collection = indexer.client.get_collection(selected_collection)
                
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
                        display_result_card(i, result_data)
                
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

