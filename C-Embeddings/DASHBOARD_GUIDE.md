# 🎨 Vector Query Dashboard - Complete Guide

## What is This?

A **beautiful, interactive web interface** for querying your ChromaDB vector collections. Ask questions in natural language and see results visualized as elegant gradient cards with rich metadata.

---

## 🚀 Quick Start

### 1. Launch the Dashboard

```bash
cd C-Embeddings/dashboard
streamlit run app.py
```

Or use the script:
```bash
./run_dashboard.sh
```

Dashboard opens at: **http://localhost:8501**

### 2. Prerequisites

Make sure you have:
- ✅ Streamlit installed (`pip install streamlit`)
- ✅ At least one indexed collection
- ✅ OpenAI API key configured (in .env)

If no collections exist:
```bash
cd ..
python run_indexer.py ../results/physics_structure.json
```

---

## 📖 Interface Overview

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  🔍 Vector Database Query Dashboard                     │
│  Explore and visualize your ChromaDB collections        │
├──────────────┬──────────────────────────────────────────┤
│   SIDEBAR    │          MAIN AREA                       │
│              │                                           │
│ ⚙️ Config    │  💬 Query Interface                      │
│              │  ┌──────────────────────────────────┐    │
│ Collection   │  │ Enter your question...           │    │
│ ▼ physics    │  └──────────────────────────────────┘    │
│              │  [🚀 Search]  [🔄 Clear]                 │
│ 📈 Stats     │                                           │
│ Vectors: 1111│  📊 Results                              │
│              │  ┌────────────────────────────────────┐  │
│ 🎛️ Settings  │  │ 🎯 Rank 1: Electromagnetic...     │  │
│ Top K: 5     │  │ ✨ 87.5%                           │  │
│              │  │ 📚 7  📄 269  🔖 0850             │  │
│ 🔧 Filters   │  │ Preview text...                    │  │
│ Chapter: 7   │  │ [📖 View Full Text]               │  │
│ Type: Exp.   │  └────────────────────────────────────┘  │
│              │  [...more results...]                    │
└──────────────┴──────────────────────────────────────────┘
```

### Sidebar Components

**1. Collection Selector**
- Dropdown with all available collections
- Shows collection name and hash
- Auto-updates when new collections added

**2. Collection Stats**
- Total vector count
- Embedding model used
- Expandable metadata details

**3. Query Settings**
- Top K slider (1-20)
- Controls number of results returned

**4. Advanced Filters** (expandable)
- Filter by chapter number
- Filter by anchor type
- Filter by page number

### Main Area Components

**1. Query Interface**
- Large text input for questions
- Search and Clear buttons
- Example queries in placeholder

**2. Results Section** (after search)
- Summary metrics (avg similarity, best match, count)
- Result cards (one per result)
- Expandable full text views

---

## 🎯 How to Use

### Basic Search Flow

**Step 1: Select Collection**
1. Open sidebar
2. Choose from dropdown
3. Check collection stats

**Step 2: Enter Question**
1. Click in search box
2. Type natural language query
3. Example: "How does electromagnetic induction work?"

**Step 3: Adjust Settings** (optional)
1. Set Top K (default: 5)
2. Add filters if needed

**Step 4: Search**
1. Click 🚀 Search button (or press Enter)
2. Wait for results (shows spinner)
3. View results below

**Step 5: Explore Results**
1. Read preview text
2. Check similarity score (color-coded)
3. Click "📖 View Full Text" to expand
4. Review metadata

### Advanced Usage

**Combine Filters**
```
Query: "motor experiments"
Filters:
  - Chapter: 6
  - Type: Experiment
  - Top K: 10
```

**Refine Search**
1. Run initial search
2. Check results
3. Adjust query or filters
4. Search again
5. Compare results

**Explore Collection**
1. Try broad queries first
2. Use filters to narrow down
3. Expand interesting results
4. Note chapter/page for reference

---

## 📊 Understanding Results

### Result Card Anatomy

```
┌────────────────────────────────────────────────┐
│ 🎯 Rank 1: Practical motors (p.239)  [✨ 87.5%]│ ← Header
├────────────────────────────────────────────────┤
│ 📚 Ch.6  📄 p.239  🔖 0840  📍 L.10891  🎯 Exp.│ ← Metadata
├────────────────────────────────────────────────┤
│ A practical motor usually uses a soft-iron...  │ ← Preview
│ [📖 View Full Text] ▼                          │ ← Expander
│  ┌──────────────────────────────────────────┐ │
│  │ [Full text content in scrollable area]   │ │
│  │ Additional metadata...                    │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

### Similarity Scores

**Score Ranges**
- **90-100%** 🟢 - Near perfect match
- **80-89%** 🟢 - Excellent match
- **70-79%** 🟢 - Very good match
- **60-69%** 🟡 - Good match
- **50-59%** 🟡 - Moderate match
- **<50%** 🔴 - Weak match

**What They Mean**
- High scores (>80%): Semantically very similar
- Medium scores (60-80%): Related but not exact
- Low scores (<60%): Tangentially related

### Metadata Fields

| Icon | Field | Meaning |
|------|-------|---------|
| 📚 | Chapter | Chapter number in document |
| 📄 | Page | Page number reference |
| 🔖 | Node ID | Unique identifier |
| 📍 | Line | Line number in source |
| 🎯 | Type | Anchor type (Experiment, Discussion, etc.) |
| 📊 | Tokens | Approximate token count |

---

## 🎨 Visual Features

### Color Coding

**Similarity Badges**
- 🟢 Green: Excellent (>80%)
- 🟡 Yellow: Good (60-80%)
- 🔴 Red: Weak (<60%)

**Gradients**
- Purple gradient: Headers and cards
- Pink gradient: Stats and metrics
- Blue gradient: Text previews

### Animations
- Card hover: Lift effect
- Button hover: Color shift
- Expand/collapse: Smooth transition
- Loading: Spinner animation

---

## 💡 Example Queries

### Conceptual Questions
```
How does electromagnetic induction work?
What is the motor effect?
Explain electric charges
Define electromagnetic force
```

### Specific Details
```
What is an armature?
How does a galvanometer work?
What happens in Experiment 7a?
```

### Procedural
```
Experiment about moving magnet and coil
How to demonstrate electromagnetic induction
Steps for motor experiment
```

### Location-Based
```
What is on page 239?
Content from chapter 7
Discussion sections about motors
```

### Topic-Based
```
Everything about motors
Electromagnetic induction experiments
DSE exam questions about charges
```

---

## 🔧 Advanced Features

### Multi-Filter Queries

**Example 1: Chapter + Type**
```
Query: "induced current"
Chapter: 7
Type: Experiment
Top K: 5
```
→ Returns only experiments from chapter 7 about induced current

**Example 2: Page + Type**
```
Query: "motor"
Page: 239
Type: Discussion
Top K: 3
```
→ Returns discussions about motors on page 239

### Adjusting Precision vs Recall

**High Precision** (fewer, more relevant)
- Top K: 3-5
- Add specific filters
- Use detailed queries

**High Recall** (more, broader)
- Top K: 10-20
- Remove filters
- Use general queries

---

## 🐛 Troubleshooting

### No Collections Found
**Problem**: Sidebar says "No collections found"  
**Solution**:
```bash
cd ..
python run_indexer.py ../results/physics_structure.json
```

### Search Returns No Results
**Problem**: Query returns 0 results  
**Causes**:
1. Filters too restrictive → Remove some filters
2. Query too specific → Try broader terms
3. Collection empty → Check collection stats

**Debug**:
```bash
python run_indexer.py --info <collection_name>
```

### Dashboard Won't Start
**Problem**: `streamlit: command not found`  
**Solution**:
```bash
pip install streamlit
```

**Problem**: Port already in use  
**Solution**: Change port in run script or:
```bash
streamlit run app.py --server.port 8502
```

### Slow Performance
**Problem**: Search takes too long  
**Solutions**:
1. Reduce Top K
2. Add filters to narrow search
3. Check collection size
4. Restart dashboard

---

## 📱 Responsive Design

### Desktop
- Full layout with sidebar
- 5-column metadata grid
- Large result cards

### Tablet
- Collapsible sidebar
- 3-4 column grid
- Medium cards

### Mobile
- Hamburger menu
- Single column
- Touch-friendly buttons

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter | Submit search |
| Ctrl/Cmd + R | Refresh page |
| Esc | Close expanders |
| Tab | Navigate fields |

---

## 🎓 Tips & Best Practices

### Writing Good Queries

✅ **Do**:
- Use natural language
- Ask complete questions
- Include context
- Be specific

❌ **Don't**:
- Use only keywords
- Write fragments
- Be overly technical
- Use exact text matches

### Interpreting Results

1. **Check similarity** first
2. **Read metadata** to verify relevance
3. **Preview text** for quick assessment
4. **Expand full text** for details
5. **Compare top results** for patterns

### Optimizing Search

1. Start broad, then refine
2. Use filters strategically
3. Adjust Top K based on need
4. Try multiple phrasings
5. Check different chapters

---

## 📊 Dashboard Performance

### Metrics

| Operation | Latency |
|-----------|---------|
| Page load | ~2 seconds |
| Collection select | <100ms |
| Search query | <500ms |
| Expand result | <50ms |
| Filter change | <100ms |

### Optimization

- Results are cached
- Indexer is cached
- Minimal re-renders
- Efficient queries

---

## 🔮 Future Plans

Coming soon:
- [ ] Dark mode toggle
- [ ] Query history
- [ ] Export results
- [ ] Highlight matches
- [ ] Compare queries
- [ ] Custom themes
- [ ] Embedding visualization

---

## 📚 Additional Resources

- **Dashboard README**: `dashboard/README.md`
- **Features List**: `dashboard/FEATURES.md`
- **Quick Start**: `dashboard/QUICKSTART.md`
- **Main Documentation**: `../README.md`

---

**Enjoy exploring your vector database! 🎉**

