# Dashboard Quick Start

## 🚀 Launch in 3 Steps

### 1. Install Streamlit
```bash
pip install streamlit
```

### 2. Make sure you have indexed data
```bash
# Check if collections exist
cd ..
python run_indexer.py --list

# If no collections, index your JSON
python run_indexer.py ../results/physics_structure.json
```

### 3. Launch Dashboard
```bash
cd dashboard
streamlit run app.py
```

Dashboard opens at: **http://localhost:8501**

---

## 🎯 Usage

1. **Select Collection** (sidebar dropdown)
2. **Set Top K** (slider, default: 5)
3. **Enter Question** (search box)
4. **Click Search** 🚀
5. **Explore Results** (click to expand full text)

---

## 💡 Example Queries

Try these:

```
How does electromagnetic induction work?
What is an armature in a motor?
Explain the motor effect
Experiment about moving magnet and coil
What happens on page 239?
```

---

## 🎨 Features

✅ Beautiful gradient UI  
✅ Color-coded similarity scores  
✅ Expandable result cards  
✅ Rich metadata display  
✅ Advanced filtering (chapter, page, type)  
✅ Real-time collection stats  

---

## 🔧 Filters

**Advanced Filters** (optional):
- Chapter: `7`
- Anchor Type: `Experiment`, `Discussion`, etc.
- Page: `239`

---

## 🎨 Similarity Colors

🟢 Green (>80%) - Excellent match  
🟡 Yellow (60-80%) - Good match  
🔴 Red (<60%) - Weak match  

---

## ⚡ Quick Tips

- Use natural language questions
- Adjust top-k for more/fewer results
- Expand cards to see full text
- Combine filters for precise results
- Check sidebar stats for collection info

---

**That's it! Start querying! 🎉**

