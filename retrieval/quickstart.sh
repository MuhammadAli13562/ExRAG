#!/bin/bash
# Quick start script for the agentic retrieval system

echo "=================================================="
echo "Agentic Retrieval System - Quick Start"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create one first:"
    echo "  python -m venv ../venv"
    echo "  source ../venv/bin/activate"
    echo "  pip install -r ../requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../venv/bin/activate

# Check if .env exists
if [ ! -f "../.env" ]; then
    echo "❌ .env file not found!"
    echo "Please create ../.env with your OPENAI_API_KEY"
    exit 1
fi

# List collections to help user choose
echo ""
echo "📚 Available Collections:"
echo ""
python run_agent.py --list-collections

echo ""
echo "=================================================="
echo "To use the system:"
echo "=================================================="
echo ""
echo "Interactive Mode:"
echo "  python run_agent.py --interactive \\"
echo "    --title-collection <TITLE_COLLECTION> \\"
echo "    --text-collection <TEXT_COLLECTION>"
echo ""
echo "Single Query:"
echo "  python run_agent.py \\"
echo "    --query \"Your question here\" \\"
echo "    --title-collection <TITLE_COLLECTION> \\"
echo "    --text-collection <TEXT_COLLECTION>"
echo ""
echo "Examples:"
echo "  python example.py"
echo ""
echo "=================================================="

