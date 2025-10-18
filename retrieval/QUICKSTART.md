# Agentic Retrieval - Quick Start

## 🎯 What You Have

An intelligent retrieval system that can:
- Search by title semantics
- Search by text content
- Explore context around relevant chunks
- Autonomously decide when and how to explore
- Synthesize comprehensive answers

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies

```bash
# Make sure you're in the virtual environment
source ../venv/bin/activate

# Install new dependencies
pip install langgraph langchain langchain-openai langchain-core
```

### 2. List Available Collections

```bash
python run_agent.py --list-collections
```

You should see collections like:
- `physics_structure_XXXXXXXX_title`
- `physics_structure_XXXXXXXX_text`

### 3. Start Querying!

**Interactive Mode** (recommended):
```bash
python run_agent.py --interactive \
  --title-collection "physics_structure_XXXXXXXX_title" \
  --text-collection "physics_structure_XXXXXXXX_text"
```

**Single Query**:
```bash
python run_agent.py \
  --query "What are the two kinds of electric charges?" \
  --title-collection "physics_structure_XXXXXXXX_title" \
  --text-collection "physics_structure_XXXXXXXX_text"
```

## 📚 Files Created

```
retrieval/
├── config.py                # Configuration settings
├── tools.py                 # Retrieval tools (search & explore)
├── agent.py                 # LangGraph agent orchestration
├── run_agent.py            # CLI interface
├── test_retrieval.py       # Test suite
├── example.py              # Usage examples
├── quickstart.sh           # Setup script
├── README.md               # Full documentation
├── IMPLEMENTATION.md       # Technical details
└── QUICKSTART.md          # This file
```

## 🧪 Test It

```bash
# Run comprehensive tests
python test_retrieval.py

# Run examples
python example.py
```

## 🔧 How It Works

```
User Query
    ↓
Agent (GPT-4o)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Title Search   │   Text Search   │  Explore Nodes  │
│   (ChromaDB)    │   (ChromaDB)    │     (JSON)      │
└─────────────────┴─────────────────┴─────────────────┘
    ↓
Context Gathering & Exploration
    ↓
Synthesized Answer
```

## 🎓 Example Session

```bash
$ python run_agent.py --interactive \
    --title-collection "physics_structure_abc123_title" \
    --text-collection "physics_structure_abc123_text"

🔍 Your query: What is the coulomb?

[Agent searches title and text collections]
[Agent finds node_0009 with high similarity]
[Agent explores nodes around 0009]
[Agent synthesizes answer]

The coulomb (C) is the SI unit of electric charge. It's a very 
large unit - when you rub an acetate strip with cloth, only about 
10^-9 C of charge is transferred. In a typical lightning flash, 
about 10 C is transferred. Smaller units like μC (10^-6 C) and 
nC (10^-9 C) are commonly used...

🔍 Your query: exit
Goodbye! 👋
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
AGENT_MODEL = "gpt-4o"          # LLM model
DEFAULT_TOP_K = 5               # Results per search
DEFAULT_EXPLORATION_COUNT = 3   # Context window
MAX_ITERATIONS = 15            # Max agent steps
```

## 📖 More Information

- **Full Documentation**: See `README.md`
- **Technical Details**: See `IMPLEMENTATION.md`
- **Complete Guide**: See `../RETRIEVAL_GUIDE.md`
- **Examples**: Run `python example.py`

## 🐛 Troubleshooting

**No collections found?**
```bash
cd ../embeddings
python run_indexer.py --list
# If empty, create embeddings:
python run_indexer.py ../results/physics_structure.json --index-field title
python run_indexer.py ../results/physics_structure.json --index-field text
```

**API errors?**
```bash
# Check your .env file
cat ../.env
# Should contain: OPENAI_API_KEY=sk-...
```

**Need help?**
```bash
python run_agent.py --help
python test_retrieval.py  # Run tests
```

## 🎉 You're Ready!

Try asking questions like:
- "Explain Coulomb's law"
- "What are the two types of charges?"
- "How do you charge objects by rubbing?"
- "What is an electric field?"

The agent will search, explore, and synthesize comprehensive answers!

