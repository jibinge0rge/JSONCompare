import json
from typing import Any, Dict, List, Tuple, Union

import orjson
import pandas as pd
import streamlit as st

JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


@st.cache_data
def try_load_json(text: str) -> Tuple[JsonType, str]:
	text = text.strip()
	if not text:
		return None, ""
	try:
		# orjson for speed and correctness
		return orjson.loads(text), ""
	except Exception:
		try:
			return json.loads(text), ""
		except Exception as e:
			return None, f"Invalid JSON: {e}"


def normalize_json(value: JsonType) -> JsonType:
	"""
	Produce a canonical, order-insensitive representation:
	- dicts: sort keys and normalize values
	- lists: sort by normalized representation
	- scalars: as-is
	"""
	if isinstance(value, dict):
		return {k: normalize_json(value[k]) for k in sorted(value.keys())}
	if isinstance(value, list):
		normalized_items = [normalize_json(v) for v in value]
		# Sort lists by their stringified canonical form for order-insensitive comparison
		try:
			return sorted(normalized_items, key=lambda x: orjson.dumps(x))
		except Exception:
			return sorted(normalized_items, key=lambda x: json.dumps(x, sort_keys=True))
	return value


@st.cache_data
def cached_normalize_json(value: JsonType) -> JsonType:
	return normalize_json(value)


def deep_equal_ignore_order(a: JsonType, b: JsonType) -> bool:
	return normalize_json(a) == normalize_json(b)


def intersect_json(a: JsonType, b: JsonType) -> JsonType:
	"""
	Compute the deepest common structure/values between a and b.
	If types differ or values differ, returns None for that branch except for lists where
	it returns the common elements (order-insensitive) based on canonical equality.
	"""
	if isinstance(a, dict) and isinstance(b, dict):
		common_keys = set(a.keys()) & set(b.keys())
		result: Dict[str, JsonType] = {}
		for k in sorted(common_keys):
			sub = intersect_json(a[k], b[k])
			if sub is not None:
				result[k] = sub
		return result if result else None

	if isinstance(a, list) and isinstance(b, list):
		# Order-insensitive intersection: match by canonical form
		canon_to_items_a: Dict[bytes, List[JsonType]] = {}
		for item in a:
			key = orjson.dumps(normalize_json(item))
			canon_to_items_a.setdefault(key, []).append(item)
		common: List[JsonType] = []
		for item in b:
			key = orjson.dumps(normalize_json(item))
			if canon_to_items_a.get(key):
				# consume one occurrence to handle duplicates correctly
				canon_to_items_a[key].pop()
				if not canon_to_items_a[key]:
					del canon_to_items_a[key]
				common.append(item)
		return common if common else None

	# Scalars or differing types
	if deep_equal_ignore_order(a, b):
		return a
	return None


@st.cache_data
def cached_intersect_json(a_json: str, b_json: str) -> JsonType:
	"""Cache intersection results based on JSON string representation"""
	a = orjson.loads(a_json)
	b = orjson.loads(b_json)
	return intersect_json(a, b)


def diff_json(a: JsonType, b: JsonType, path: str = "$") -> Dict[str, Any]:
	"""
	Return a structured diff with:
	- only_in_a: list of paths and values
	- only_in_b: list of paths and values
	- modified: list of paths with (a_value, b_value)
	Uses order-insensitive comparison for dicts and lists.
	"""
	diffs = {"only_in_a": [], "only_in_b": [], "modified": []}

	if isinstance(a, dict) and isinstance(b, dict):
		keys_a, keys_b = set(a.keys()), set(b.keys())
		for k in sorted(keys_a - keys_b):
			diffs["only_in_a"].append({"path": f"{path}.{k}", "value": a[k]})
		for k in sorted(keys_b - keys_a):
			diffs["only_in_b"].append({"path": f"{path}.{k}", "value": b[k]})
		for k in sorted(keys_a & keys_b):
			sub_a, sub_b = a[k], b[k]
			if isinstance(sub_a, (dict, list)) or isinstance(sub_b, (dict, list)):
				sub_diffs = diff_json(sub_a, sub_b, f"{path}.{k}")
				for key in ("only_in_a", "only_in_b", "modified"):
					diffs[key].extend(sub_diffs[key])
			else:
				if not deep_equal_ignore_order(sub_a, sub_b):
					diffs["modified"].append({
						"path": f"{path}.{k}",
						"a": sub_a,
						"b": sub_b,
					})
		return diffs

	if isinstance(a, list) and isinstance(b, list):
		# Compare as multisets based on canonical representation
		def multiset_counts(lst: List[JsonType]) -> Dict[bytes, int]:
			counts: Dict[bytes, int] = {}
			for item in lst:
				key = orjson.dumps(normalize_json(item))
				counts[key] = counts.get(key, 0) + 1
			return counts

		counts_a = multiset_counts(a)
		counts_b = multiset_counts(b)

		# Items only in a
		for key, cnt in counts_a.items():
			extra = cnt - counts_b.get(key, 0)
			if extra > 0:
				try:
					value = orjson.loads(key)
				except Exception:
					value = json.loads(key)
				diffs["only_in_a"].append({"path": path + "[]", "value": value, "count": extra})

		# Items only in b
		for key, cnt in counts_b.items():
			extra = cnt - counts_a.get(key, 0)
			if extra > 0:
				try:
					value = orjson.loads(key)
				except Exception:
					value = json.loads(key)
				diffs["only_in_b"].append({"path": path + "[]", "value": value, "count": extra})

		# No modified entries at list level; modifications are caught inside dict elements
		return diffs

	# Different types or scalars
	if not deep_equal_ignore_order(a, b):
		diffs["modified"].append({"path": path, "a": a, "b": b})
	return diffs


@st.cache_data
def cached_diff_json(a_json: str, b_json: str) -> Dict[str, Any]:
	"""Cache diff results based on JSON string representation"""
	a = orjson.loads(a_json)
	b = orjson.loads(b_json)
	return diff_json(a, b)


def similarity_score(a: JsonType, b: JsonType) -> float:
	"""
	Rough similarity score based on normalized serialization overlap.
	1.0 means identical under order-insensitive rules.
	"""
	ca, cb = normalize_json(a), normalize_json(b)
	if ca == cb:
		return 1.0
	try:
		sa = orjson.dumps(ca)
		sb = orjson.dumps(cb)
	except Exception:
		sa = json.dumps(ca, sort_keys=True).encode()
		sb = json.dumps(cb, sort_keys=True).encode()
	# Jaccard on byte bigrams as a cheap proxy
	def ngrams(s: bytes, n: int = 2) -> set:
		return {s[i:i+n] for i in range(max(1, len(s) - n + 1))}

	ga, gb = ngrams(sa), ngrams(sb)
	if not ga and not gb:
		return 1.0
	if not ga or not gb:
		return 0.0
	inter = len(ga & gb)
	union = len(ga | gb)
	return inter / union if union else 0.0


@st.cache_data
def cached_similarity_score(a_json: str, b_json: str) -> float:
	"""Cache similarity score based on JSON string representation"""
	a = orjson.loads(a_json)
	b = orjson.loads(b_json)
	return similarity_score(a, b)


def format_json_value(value: Any, max_length: int = 100) -> str:
	"""Format JSON value for display in table"""
	try:
		json_str = json.dumps(value, ensure_ascii=False)
		if len(json_str) > max_length:
			return json_str[:max_length] + "..."
		return json_str
	except Exception:
		return str(value)[:max_length]


def create_diff_table(diffs: Dict[str, Any], search_filter: str = "") -> pd.DataFrame:
	"""Create a pandas DataFrame for the differences table"""
	rows = []
	
	only_a = diffs.get("only_in_a", [])
	only_b = diffs.get("only_in_b", [])
	modified = diffs.get("modified", [])
	
	for item in only_a:
		path = item["path"]
		count = item.get("count", 1)
		value_str = format_json_value(item["value"])
		if count > 1:
			path = f"{path} (×{count})"
		if not search_filter or search_filter.lower() in (path + value_str).lower():
			rows.append({
				"Path": path,
				"In Base": value_str,
				"In Modified": "",
				"Status": "Only in Base"
			})
	
	for item in only_b:
		path = item["path"]
		count = item.get("count", 1)
		value_str = format_json_value(item["value"])
		if count > 1:
			path = f"{path} (×{count})"
		if not search_filter or search_filter.lower() in (path + value_str).lower():
			rows.append({
				"Path": path,
				"In Base": "",
				"In Modified": value_str,
				"Status": "Only in Modified"
			})
	
	for item in modified:
		path = item["path"]
		value_a_str = format_json_value(item["a"])
		value_b_str = format_json_value(item["b"])
		if not search_filter or search_filter.lower() in (path + value_a_str + value_b_str).lower():
			rows.append({
				"Path": path,
				"In Base": value_a_str,
				"In Modified": value_b_str,
				"Status": "Modified"
			})
	
	return pd.DataFrame(rows)


def style_diff_table(df: pd.DataFrame):
	"""Apply styling to the differences table"""
	def row_style(row):
		styles = {
			"Only in Base": {"background-color": "#ef4444"},
			"Only in Modified": {"background-color": "#22c55e"},
			"Modified": {"background-color": "#eab308"}
		}
		color = styles.get(row["Status"], {}).get("background-color", "transparent")
		return [f"background-color: {color}"] * len(row)
	
	if df.empty:
		return df
	return df.style.apply(row_style, axis=1)


def render_summary(diffs: Dict[str, Any], common: JsonType, similarity: float):
	"""Render summary panel with counts"""
	st.markdown("### Differences Summary")
	st.markdown("---")
	
	only_a_count = len(diffs.get("only_in_a", []))
	only_b_count = len(diffs.get("only_in_b", []))
	modified_count = len(diffs.get("modified", []))
	
	# Count common elements
	common_count = 0
	if common is not None:
		def count_elements(obj):
			if isinstance(obj, dict):
				return 1 + sum(count_elements(v) for v in obj.values())
			elif isinstance(obj, list):
				return len(obj) + sum(count_elements(item) for item in obj)
			return 1
		common_count = count_elements(common)
	
	col1, col2, col3, col4, col5 = st.columns(5)
	with col1:
		st.markdown(f"**Only in Base**\n\n{only_a_count}")
	with col2:
		st.markdown(f"**Only in Modified**\n\n{only_b_count}")
	with col3:
		st.markdown(f"**Modified**\n\n{modified_count}")
	with col4:
		st.markdown(f"**Common**\n\n{common_count}")
	with col5:
		st.markdown(f"**Similarity**\n\n{similarity:.3f}")
	
	st.markdown("")


def render_metrics(a: JsonType, b: JsonType):
	ca, cb = normalize_json(a), normalize_json(b)
	score = similarity_score(a, b)
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Similarity (0-1)", f"{score:.3f}")
	with col2:
		st.metric("Size A (chars)", f"{len(orjson.dumps(ca))}")
	with col3:
		st.metric("Size B (chars)", f"{len(orjson.dumps(cb))}")


def render_diff(diffs: Dict[str, Any], show_only_diffs: bool = False, search_filter: str = ""):
	"""Render differences with table view and optional filtering"""
	only_a = diffs.get("only_in_a", [])
	only_b = diffs.get("only_in_b", [])
	modified = diffs.get("modified", [])

	empty = not (only_a or only_b or modified)
	if empty:
		st.success("No differences. JSONs are equivalent (ignoring order).")
		return

	# Create and display table
	df = create_diff_table(diffs, search_filter)
	
	if df.empty and search_filter:
		st.info(f"No differences match the search filter: '{search_filter}'")
		return
	
	if not df.empty:
		st.markdown("### Differences Table")
		styled_df = style_diff_table(df)
		st.dataframe(
			styled_df,
			use_container_width=True,
			hide_index=True,
			height=400
		)
		st.markdown("")
	
	# Detailed view
	if not show_only_diffs:
		st.markdown("### Detailed View")
		col_a, col_m, col_b = st.columns([1, 1, 1])
		with col_a:
			st.markdown("**Only in Base**")
			if not only_a:
				st.caption("None")
			else:
				for item in only_a:
					count = item.get("count")
					suffix = f" ×{count}" if count else ""
					st.markdown(f"`{item['path']}`{suffix}")
					st.code(json.dumps(item.get("value"), indent=2, ensure_ascii=False), language="json")
		with col_m:
			st.markdown("**Modified**")
			if not modified:
				st.caption("None")
			else:
				for item in modified:
					st.markdown(f"`{item['path']}`")
					cols = st.columns(2)
					with cols[0]:
						st.caption("Base")
						st.code(json.dumps(item.get("a"), indent=2, ensure_ascii=False), language="json")
					with cols[1]:
						st.caption("Modified")
						st.code(json.dumps(item.get("b"), indent=2, ensure_ascii=False), language="json")
		with col_b:
			st.markdown("**Only in Modified**")
			if not only_b:
				st.caption("None")
			else:
				for item in only_b:
					count = item.get("count")
					suffix = f" ×{count}" if count else ""
					st.markdown(f"`{item['path']}`{suffix}")
					st.code(json.dumps(item.get("value"), indent=2, ensure_ascii=False), language="json")


def render_common(common: JsonType, expanded: bool = False):
	if common is None:
		st.info("No common structure/values found.")
		return
	st.markdown("### Common Structure/Values")
	# Expand/Collapse buttons
	if "common_expanded" not in st.session_state:
		st.session_state.common_expanded = False
	
	col_expand, col_space = st.columns([1, 5])
	with col_expand:
		if st.button("Expand All", key="btn_expand_common"):
			st.session_state.common_expanded = True
		if st.button("Collapse All", key="btn_collapse_common"):
			st.session_state.common_expanded = False
	
	expanded = st.session_state.common_expanded
	st.json(common, expanded=expanded)


def main():
	st.set_page_config(page_title="JSON Compare (Order-Insensitive)", page_icon=None, layout="wide")
	
	# Enhanced styling
	st.markdown(
		"""
		<style>
			/* Enhanced UI styling */
			section.main > div { padding-top: 0.5rem; }
			textarea { 
				font-family: 'Menlo', 'Consolas', 'Source Code Pro', ui-monospace, SFMono-Regular, Monaco, "Liberation Mono", "Courier New", monospace;
				border: 1px solid #e0e0e0;
				border-radius: 4px;
				padding: 8px;
				box-shadow: 0 1px 3px rgba(0,0,0,0.1);
			}
			.stTextArea > div > div > textarea {
				font-family: 'Menlo', 'Consolas', 'Source Code Pro', ui-monospace, SFMono-Regular, Monaco, "Liberation Mono", "Courier New", monospace;
			}
			/* Section spacing */
			.element-container {
				margin-bottom: 1rem;
			}
			/* Code block styling */
			pre {
				border: 1px solid #e0e0e0;
				border-radius: 4px;
				padding: 12px;
				background-color: #f8f9fa;
				box-shadow: 0 1px 2px rgba(0,0,0,0.05);
			}
			/* Summary panel styling */
			[data-testid="stMetricValue"] {
				font-size: 1.2rem;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)
	
	st.title("JSON Compare (Order-Insensitive)")
	st.caption("Keys and sub-keys in any order. Lists compared as multisets.")

	with st.sidebar:
		st.header("Options")
		show_raw = st.toggle("Show normalized JSON", value=False, help="Display canonicalized versions for clarity.")
		show_only_diffs = st.toggle("Show only differences", value=False, help="Hide detailed view in Differences tab.")

	col_left, col_right = st.columns(2)
	
	default_a = """{
  "name": "Alice",
  "roles": ["user", "admin"],
  "meta": {"age": 30, "active": true},
  "items": [
    {"id": 1, "qty": 2},
    {"id": 2, "qty": 1}
  ]
}"""
	
	default_b = """{
  "roles": ["admin", "user"],
  "name": "Alice",
  "meta": {"active": true, "age": 30},
  "items": [
    {"qty": 1, "id": 2},
    {"id": 3, "qty": 5}
  ],
  "extra": "surprise"
}"""
	
	# Initialize session state for text areas if not exists
	if "text_a" not in st.session_state:
		st.session_state.text_a = default_a
	if "text_b" not in st.session_state:
		st.session_state.text_b = default_b
	
	with col_left:
		st.subheader("Base JSON")
		text_a = st.text_area("", height=320, value=st.session_state.text_a, key="text_a_input")
		st.session_state.text_a = text_a
	
	with col_right:
		st.subheader("Modified JSON")
		text_b = st.text_area("", height=320, value=st.session_state.text_b, key="text_b_input")
		st.session_state.text_b = text_b

	# Compare button
	compare_clicked = st.button("Compare", type="primary", use_container_width=False)
	
	# Only process when compare button is clicked
	if compare_clicked:
		a, err_a = try_load_json(text_a)
		b, err_b = try_load_json(text_b)
		
		if err_a or err_b:
			st.error(err_a or err_b)
			return

		if a is None or b is None:
			st.info("Provide both JSON inputs to compare.")
			return

		# Cache results using JSON string representation
		a_json_str = orjson.dumps(a).decode()
		b_json_str = orjson.dumps(b).decode()
		
		# Calculate metrics and diffs
		diffs = cached_diff_json(a_json_str, b_json_str)
		common = cached_intersect_json(a_json_str, b_json_str)
		similarity = cached_similarity_score(a_json_str, b_json_str)
		
		# Store in session state for use across tabs
		st.session_state.last_compare = True
		st.session_state.diffs = diffs
		st.session_state.common = common
		st.session_state.similarity = similarity
		st.session_state.json_a = a
		st.session_state.json_b = b
	elif "last_compare" in st.session_state and st.session_state.last_compare:
		# Show previous results if they exist
		diffs = st.session_state.diffs
		common = st.session_state.common
		similarity = st.session_state.similarity
		a = st.session_state.json_a
		b = st.session_state.json_b
	else:
		# First time - show prompt
		st.info("Enter JSON inputs and click 'Compare' to see results.")
		return

	tab_overview, tab_common, tab_diff, tab_raw = st.tabs(["Overview", "Common", "Differences", "Raw"])

	with tab_overview:
		render_metrics(a, b)
		st.markdown("\n")
		st.write("Use the tabs to explore common structure and differences.")

	with tab_common:
		render_common(common)

	with tab_diff:
		# Search box
		search_filter = st.text_input("Search differences", placeholder="Filter by path or value...", key="search_diff")
		
		# Render summary
		render_summary(diffs, common, similarity)
		
		# Render differences
		render_diff(diffs, show_only_diffs, search_filter)

	with tab_raw:
		# Expand/Collapse buttons
		if "expand_state" not in st.session_state:
			st.session_state.expand_state = False
		
		col_expand, col_space = st.columns([1, 5])
		with col_expand:
			if st.button("Expand All", key="expand_all"):
				st.session_state.expand_state = True
			if st.button("Collapse All", key="collapse_all"):
				st.session_state.expand_state = False
		
		expanded = st.session_state.expand_state
		
		cols = st.columns(2)
		with cols[0]:
			st.caption("Base (input)")
			st.json(a, expanded=expanded)
		with cols[1]:
			st.caption("Modified (input)")
			st.json(b, expanded=expanded)
		if show_raw:
			st.divider()
			cols2 = st.columns(2)
			with cols2[0]:
				st.caption("Base (normalized)")
				st.json(normalize_json(a), expanded=expanded)
			with cols2[1]:
				st.caption("Modified (normalized)")
				st.json(normalize_json(b), expanded=expanded)


if __name__ == "__main__":
	main()
