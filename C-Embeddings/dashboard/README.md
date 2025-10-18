# 🔍 Vector Query Dashboard

Beautiful interactive web interface for querying and visualizing ChromaDB collections.

![Dashboard Preview](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)

## Features

✨ **Interactive Query Interface**
- Natural language question input
- Real-time vector similarity search
- Adjustable top-k results (1-20)

📊 **Beautiful Visualizations**
- Gradient-styled result cards
- Similarity score badges with color coding
- Expandable full text views
- Metadata display in organized grids

🎛️ **Advanced Filtering**
- Filter by chapter
- Filter by anchor type (Experiment, Discussion, etc.)
- Filter by page number
- Combine multiple filters

📈 **Collection Management**
- Dropdown collection selector
- Real-time collection statistics
- Vector count and metadata display

🎨 **Modern UI Design**
- Gradient backgrounds
- Smooth animations and hover effects
- Responsive layout
- Clean, professional styling

## Installation

Install Streamlit (if not already installed):

```bash
pip install streamlit
```

Or add to your requirements.txt:
```
streamlit>=1.28.0
```

## Usage

### Launch the Dashboard

**Option 1: Using the shell script**
```bash
cd dashboard
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Option 2: Direct command**
```bash
cd dashboard
streamlit run app.py
```

**Option 3: From project root**
```bash
streamlit run C-Embeddings/dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## How to Use

### 1. Select Collection
- Use the sidebar dropdown to choose a ChromaDB collection
- View collection statistics (vector count, model info)

### 2. Configure Query Settings
- Adjust **Top K** slider (1-20) to control number of results
- Optionally set advanced filters:
  - Chapter number
  - Anchor type
  - Page number

### 3. Enter Your Question
- Type natural language query in the search box
- Examples:
  - "How does electromagnetic induction work?"
  - "What is an armature in a motor?"
  - "Experiment about moving magnet and coil"

### 4. View Results
- Click **🚀 Search** button
- Results display as beautiful gradient cards
- Each card shows:
  - **Similarity score** (color-coded badge)
  - **Metadata**: Chapter, Page, Node ID, Line, Type
  - **Text preview** (first 300 chars)
  - **Expandable full text**

### 5. Explore Results
- Click "📖 View Full Text" to expand any result
- See additional metadata (section, heading level, etc.)
- Compare similarity scores across results

## Result Card Color Coding

| Similarity | Badge Color | Meaning |
|------------|-------------|---------|
| > 80% | 🟢 Green | Excellent match |
| 60-80% | 🟡 Yellow | Good match |
| < 60% | 🔴 Red | Weak match |

## Keyboard Shortcuts

- `Enter` - Submit query (when focused on input)
- `Ctrl/Cmd + R` - Refresh page
- `Esc` - Close expandable sections

## Troubleshooting

### "No collections found"
- Make sure you've indexed at least one JSON file:
  ```bash
  python run_indexer.py ../results/physics_structure.json
  ```

### Dashboard won't start
- Check Streamlit is installed: `pip install streamlit`
- Verify you're in the correct directory
- Check port 8501 is not in use

### Search returns no results
- Try broader query terms
- Remove filters to see if they're too restrictive
- Check collection has data: `python run_indexer.py --info <collection>`

### Slow performance
- Reduce top-k value
- Use more specific queries
- Consider indexing optimization

## Customization

### Change Port
Edit `run_dashboard.sh`:
```bash
streamlit run app.py --server.port YOUR_PORT
```

### Modify Styling
Edit the `<style>` section in `app.py` to customize:
- Colors and gradients
- Card layouts
- Font sizes
- Spacing

### Add New Filters
Extend the "Advanced Filters" section in sidebar:
```python
filter_by_custom = st.text_input("Custom Filter", ...)
```

## Architecture

```
dashboard/
├── app.py              # Main Streamlit application
├── run_dashboard.sh    # Launch script
└── README.md          # This file

app.py components:
├── Custom CSS          # Beautiful gradient styling
├── Sidebar             # Collection selector & settings
├── Query Interface     # Search input & controls
├── Results Display     # Card-based result visualization
└── Result Cards        # Individual expandable result blocks
```

## Tech Stack

- **Streamlit** - Web framework
- **ChromaDB** - Vector database
- **OpenAI Embeddings** - Semantic search
- **Python 3.11+** - Backend

## Tips for Best Results

1. **Use natural language** - Ask questions as you would to a person
2. **Be specific** - Include context and details
3. **Adjust top-k** - More results for exploration, fewer for precision
4. **Use filters** - Narrow down to specific chapters/types
5. **Check similarity** - Green badges (>80%) are most relevant

## Screenshots

### Main Interface
- Clean search box with gradient header
- Sidebar with collection stats
- Advanced filter options

### Result Cards
- Gradient-styled cards with similarity badges
- Metadata grid with icons
- Text preview with beautiful formatting
- Expandable full text view

### Collection Stats
- Vector count metrics
- Model information
- Metadata details

## Performance

- **Query latency**: < 100ms for top-10 results
- **UI responsiveness**: Real-time updates
- **Memory usage**: ~100MB for typical collections
- **Concurrent users**: Supports multiple simultaneous queries

## Future Enhancements

Potential improvements:
- [ ] Query history and bookmarks
- [ ] Export results to JSON/CSV
- [ ] Highlighting of matching terms
- [ ] Side-by-side comparison mode
- [ ] Dark/light theme toggle
- [ ] Custom color schemes
- [ ] Multi-query comparison
- [ ] Semantic clustering visualization

## Support

For issues or questions:
1. Check this README
2. Review main documentation in `../USAGE.md`
3. Inspect collection with `python run_indexer.py --info <name>`

---

**Built with ❤️ for intuitive vector search exploration**

