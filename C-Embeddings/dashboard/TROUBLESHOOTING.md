# Dashboard Troubleshooting Guide

## ✅ Fixed: Import Error

### The Problem
If you see an error like:
```
ModuleNotFoundError: No module named 'indexer'
```

### The Solution
The dashboard must be run from the **C-Embeddings** directory (parent of dashboard), not from within the dashboard directory itself.

### ✅ Correct Ways to Run

**Option 1: Using the shell script (Mac/Linux)**
```bash
cd /Users/ali-98/data/projects/PageIndex/C-Embeddings
./dashboard/run_dashboard.sh
```

**Option 2: Using the Python launcher (Cross-platform)**
```bash
cd /Users/ali-98/data/projects/PageIndex/C-Embeddings
python run_dashboard.py
```

**Option 3: Direct streamlit command**
```bash
cd /Users/ali-98/data/projects/PageIndex/C-Embeddings
streamlit run dashboard/app.py
```

### ❌ Wrong Way (Don't do this)
```bash
cd /Users/ali-98/data/projects/PageIndex/C-Embeddings/dashboard
streamlit run app.py  # ❌ This will fail!
```

## Why This Matters

The dashboard needs to import:
- `indexer.py` 
- `config.py`
- `metadata_extractor.py`
- `embedder.py`

All these modules are in the `C-Embeddings/` directory. Running from `C-Embeddings/dashboard/` breaks the import paths.

## Additional Troubleshooting

### Check your working directory
```bash
pwd  # Should show: .../PageIndex/C-Embeddings
```

### Verify the structure
```
C-Embeddings/
├── indexer.py          ← Needs to be importable
├── config.py           ← Needs to be importable
├── embedder.py         ← Needs to be importable
├── metadata_extractor.py
├── run_dashboard.py    ← Run this
└── dashboard/
    ├── app.py
    └── run_dashboard.sh
```

### Still having issues?
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that `OPENAI_API_KEY` is set in your `.env` file
3. Ensure you have indexed at least one JSON file before running the dashboard

