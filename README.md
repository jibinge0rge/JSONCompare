## JSONCompare - Streamlit App

A focused, beautiful JSON comparison tool in Streamlit. It compares two JSON documents order-insensitively:
- Keys and nested keys can be in any order
- Lists are compared as multisets (order-insensitive), including lists of objects
- Shows common structure/values, differences unique to each side, and modified values

### What changed
- Copy-paste only (no file uploads)
- New tabbed UI: Overview, Common, Differences, Raw
- Styled badges and improved layout for clarity

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open the URL shown (usually `http://localhost:8501`).

### Usage
- Paste JSON A and JSON B into the editors.
- Use tabs to see metrics, common parts, and differences.
- Toggle "Show normalized JSON" in the sidebar to display canonical views.

### Notes
- The comparison is order-insensitive for dicts and lists.
- Duplicates in lists are handled as counts.
- If there is nothing in common, the app will state that clearly.
