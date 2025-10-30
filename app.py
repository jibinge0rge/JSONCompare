import json
from typing import Any, Dict, List, Tuple, Union

import orjson
import streamlit as st

JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


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


def badge(text: str, kind: str) -> str:
	colors = {
		"ok": "#0e8f4b",
		"warn": "#c97b00",
		"err": "#b00020",
		"muted": "#5f6368",
	}
	bg = colors.get(kind, "#5f6368")
	return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:{bg};color:white;font-size:12px;margin-right:6px'>{text}</span>"


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


def render_diff(diffs: Dict[str, Any]):
	st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
	only_a = diffs.get("only_in_a", [])
	only_b = diffs.get("only_in_b", [])
	modified = diffs.get("modified", [])

	empty = not (only_a or only_b or modified)
	if empty:
		st.success("No differences. JSONs are equivalent (ignoring order).")
		return

	col_a, col_m, col_b = st.columns([1, 1, 1])
	with col_a:
		st.markdown(badge("Only in A", "warn"), unsafe_allow_html=True)
		if not only_a:
			st.caption("None")
		else:
			for item in only_a:
				count = item.get("count")
				suffix = f" ×{count}" if count else ""
				st.markdown(f"`{item['path']}`{suffix}")
				st.code(json.dumps(item.get("value"), indent=2, ensure_ascii=False), language="json")
	with col_m:
		st.markdown(badge("Modified", "err"), unsafe_allow_html=True)
		if not modified:
			st.caption("None")
		else:
			for item in modified:
				st.markdown(f"`{item['path']}`")
				cols = st.columns(2)
				with cols[0]:
					st.caption("A")
					st.code(json.dumps(item.get("a"), indent=2, ensure_ascii=False), language="json")
				with cols[1]:
					st.caption("B")
					st.code(json.dumps(item.get("b"), indent=2, ensure_ascii=False), language="json")
	with col_b:
		st.markdown(badge("Only in B", "warn"), unsafe_allow_html=True)
		if not only_b:
			st.caption("None")
		else:
			for item in only_b:
				count = item.get("count")
				suffix = f" ×{count}" if count else ""
				st.markdown(f"`{item['path']}`{suffix}")
				st.code(json.dumps(item.get("value"), indent=2, ensure_ascii=False), language="json")


def render_common(common: JsonType):
	st.markdown(badge("Common", "ok"), unsafe_allow_html=True)
	if common is None:
		st.info("No common structure/values found.")
		return
	st.json(common, expanded=False)


def main():
	st.set_page_config(page_title="JSON Compare (Order-Insensitive)", page_icon="🧩", layout="wide")
	st.markdown(
		"""
		<style>
			/* Subtle UI polish */
			section.main > div { padding-top: 0.5rem; }
			textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
		</style>
		""",
		unsafe_allow_html=True,
	)
	st.title("🧩 JSON Compare (Order-Insensitive)")
	st.caption("Keys and sub-keys in any order. Lists compared as multisets.")

	with st.sidebar:
		st.header("Options")
		show_raw = st.toggle("Show normalized JSON", value=False, help="Display canonicalized versions for clarity.")
		st.divider()

	col_left, col_right = st.columns(2)
	with col_left:
		st.subheader("JSON A")
		default_a = """
{
  "name": "Alice",
  "roles": ["user", "admin"],
  "meta": {"age": 30, "active": true},
  "items": [
    {"id": 1, "qty": 2},
    {"id": 2, "qty": 1}
  ]
}
""".strip()
		text_a = st.text_area("", height=320, value=default_a, key="text_a")
	with col_right:
		st.subheader("JSON B")
		default_b = """
{
  "roles": ["admin", "user"],
  "name": "Alice",
  "meta": {"active": true, "age": 30},
  "items": [
    {"qty": 1, "id": 2},
    {"id": 3, "qty": 5}
  ],
  "extra": "surprise"
}
""".strip()
		text_b = st.text_area("", height=320, value=default_b, key="text_b")

	a, err_a = try_load_json(text_a)
	b, err_b = try_load_json(text_b)

	if err_a or err_b:
		st.error(err_a or err_b)
		return

	if a is None or b is None:
		st.info("Provide both JSON inputs to compare.")
		return

	tab_overview, tab_common, tab_diff, tab_raw = st.tabs(["Overview", "Common", "Differences", "Raw"])

	with tab_overview:
		render_metrics(a, b)
		st.markdown("\n")
		st.write("Use the tabs to explore common structure and differences.")

	with tab_common:
		render_common(intersect_json(a, b))

	with tab_diff:
		render_diff(diff_json(a, b))

	with tab_raw:
		cols = st.columns(2)
		with cols[0]:
			st.caption("A (input)")
			st.code(json.dumps(try_load_json(text_a)[0], indent=2, ensure_ascii=False), language="json")
		with cols[1]:
			st.caption("B (input)")
			st.code(json.dumps(try_load_json(text_b)[0], indent=2, ensure_ascii=False), language="json")
		if show_raw:
			st.divider()
			cols2 = st.columns(2)
			with cols2[0]:
				st.caption("A (normalized)")
				st.code(json.dumps(normalize_json(a), indent=2, ensure_ascii=False), language="json")
			with cols2[1]:
				st.caption("B (normalized)")
				st.code(json.dumps(normalize_json(b), indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
	main()
