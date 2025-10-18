#!/bin/bash
# Launch the Vector Query Dashboard

# Change to C-Embeddings directory (parent of dashboard)
cd "$(dirname "$0")/.."

# Run streamlit from the parent directory with the correct module path
streamlit run dashboard/app.py --server.port 8501 --server.address localhost

