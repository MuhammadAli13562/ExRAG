#!/bin/bash
# Launch the Vector Query Dashboard

# Change to project root (two levels up from dashboard)
cd "$(dirname "$0")/../.."

# Run streamlit from the project root with the correct module path
streamlit run embeddings/dashboard/app.py --server.port 8501 --server.address localhost

