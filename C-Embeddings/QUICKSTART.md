# 🚀 Quick Start Guide

## Running the Dashboard

### Method 1: Python Launcher (Recommended)
```bash
cd /Users/ali-98/data/projects/PageIndex/C-Embeddings
python run_dashboard.py
```

### Method 2: Shell Script (Mac/Linux)
```bash
cd /Users/ali-98/data/projects/PageIndex/C-Embeddings
./dashboard/run_dashboard.sh
```

### Method 3: Direct Streamlit
```bash
cd /Users/ali-98/data/projects/PageIndex/C-Embeddings
streamlit run dashboard/app.py
```

## ⚠️ Important
Always run from the `C-Embeddings/` directory, not from `C-Embeddings/dashboard/`!

## Before Running
1. Make sure you have indexed at least one JSON file:
   ```bash
   python run_indexer.py path/to/your_structure.json
   ```

2. Ensure your `.env` file has the OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_key_here
   ```

## Access the Dashboard
Once running, open your browser to:
```
http://localhost:8501
```

## Troubleshooting
If you encounter import errors, see: `dashboard/TROUBLESHOOTING.md`

