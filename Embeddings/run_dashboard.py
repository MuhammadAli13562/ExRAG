#!/usr/bin/env python3
"""
Launch the Vector Query Dashboard

This script ensures the dashboard is run from the correct directory
with proper module paths set up.
"""
import sys
import subprocess
from pathlib import Path

def main():
    # Change to project root (parent of embeddings directory)
    project_root = Path(__file__).parent.parent
    
    # Run streamlit from the project root
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "embeddings/dashboard/app.py",
        "--server.port",
        "8501",
        "--server.address",
        "localhost"
    ]
    
    print(f"🚀 Starting dashboard from: {project_root}")
    print(f"📂 Running command: {' '.join(cmd)}")
    print(f"🌐 Dashboard will be available at: http://localhost:8501")
    print()
    
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

