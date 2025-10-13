import argparse
import ast
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple, Union
import pandas as pd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read a CSV, replace the IDs in edge_list per row with the corresponding node's attr from node_list,
and insert the result into a new column right after edge_list named "sematic edge list". By default saves in-place (creates a .bak backup first).
"""


def parse_literal(cell: Any) -> Any:
    """
    Try to parse a cell value into a Python object.
    Attempts in order: already an object -> return; ast.literal_eval -> return; json.loads -> return; otherwise return an empty list.
    """
    if isinstance(cell, (list, dict, tuple)):
        return cell
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    if not isinstance(cell, str):
        return cell

    s = cell.strip()
    if not s:
        return []

    # Prefer Python literal parsing (original data looks like this)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # Fallback to JSON parsing
    try:
        return json.loads(s)
    except Exception:
        return []


def build_id2attr(node_list_obj: Any) -> Dict[Union[int, str], str]:
    """
    node_list is expected in the form:
    [[1, {'name': 'ingredient', 'attr': 'Garlic, raw'}], [2, {...}], ...]
    Return a mapping from id to attr.
    """
    id2attr: Dict[Union[int, str], str] = {}
    if not isinstance(node_list_obj, (list, tuple)):
        return id2attr

    for item in node_list_obj:
        try:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            node_id = item[0]
            info = item[1]
            if isinstance(info, dict):
                attr_val = info.get("attr", None)
            else:
                attr_val = None
            if attr_val is None:
                # Fallback: if no attr, use stringified info as the value
                attr_val = str(info)
            id2attr[node_id] = str(attr_val)
        except Exception:
            # On error skip the node
            continue
    return id2attr


def convert_edges(edge_list_obj: Any, id2attr: Dict[Union[int, str], str]) -> List[List[Any]]:
    """
    edge_list is expected in the form:
    [[src_id, 'relation', tgt_id], ...]
    Replace src_id/tgt_id with the corresponding attr from id2attr (if not found, keep the original value as a string).
    """
    out: List[List[Any]] = []
    if not isinstance(edge_list_obj, (list, tuple)):
        return out

    for e in edge_list_obj:
        try:
            if not isinstance(e, (list, tuple)) or len(e) < 3:
                continue
            src, rel, tgt = e[0], e[1], e[2]
            src_name = id2attr.get(src, str(src))
            tgt_name = id2attr.get(tgt, str(tgt))
            out.append([src_name, rel, tgt_name])
        except Exception:
            # On error continue
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description="Replace IDs in CSV edge_list with corresponding node_list attrs and write back to CSV.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (if not provided, overwrite input file in-place and create a .bak backup)",
    )
    parser.add_argument(
        "--node-col",
        default="node_list",
        help="Column name for node_list (default: node_list)",
    )
    parser.add_argument(
        "--edge-col",
        default="edge_list",
        help="Column name for edge_list (default: edge_list)",
    )
    parser.add_argument(
        "--new-col",
        default="sematic edge list",
        help="New column name (default: sematic edge list)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Only process the first N rows; by default process all.",
    )
    args = parser.parse_args()

    input_csv = args.csv
    output_csv = args.output

    if not os.path.isfile(input_csv):
        print(f"CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_csv, dtype=str, keep_default_na=False, low_memory=False)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.edge_col not in df.columns or args.node_col not in df.columns:
        print(
            f"Missing required columns. Found: {list(df.columns)}; need: {args.node_col}, {args.edge_col}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Process rows one by one (optional limit)
    results: List[str] = []
    processed = 0
    limit = None if args.limit is None else max(0, args.limit)

    for _, row in df.iterrows():
        if limit is not None and processed >= limit:
            results.append("")  # leave unprocessed rows empty
            continue

        node_cell = row.get(args.node_col, "")
        edge_cell = row.get(args.edge_col, "")

        node_list_obj = parse_literal(node_cell)
        edge_list_obj = parse_literal(edge_cell)

        id2attr = build_id2attr(node_list_obj)
        converted = convert_edges(edge_list_obj, id2attr)

        # To preserve original style, write back using Python literal format (avoids JSON escaping)
        results.append(repr(converted))
        processed += 1

    # Insert after the edge_list column
    edge_idx = df.columns.get_loc(args.edge_col)
    # If a column with the same name exists, remove it first
    if args.new_col in df.columns:
        df.drop(columns=[args.new_col], inplace=True)
    df.insert(edge_idx + 1, args.new_col, results)

    # Output
    if output_csv is None:
        # Overwrite in-place, backup first
        backup_path = input_csv + ".bak"
        try:
            shutil.copyfile(input_csv, backup_path)
        except Exception as e:
            print(f"Failed to create backup: {e}", file=sys.stderr)
            sys.exit(1)
        try:
            df.to_csv(input_csv, index=False)
        except Exception as e:
            print(f"Failed to write CSV: {e}. Your original file is backed up at {backup_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Done. Updated file saved in-place. Processed: {processed}/{len(df)}. Backup: {backup_path}")
    else:
        try:
            df.to_csv(output_csv, index=False)
        except Exception as e:
            print(f"Failed to write output CSV: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Done. Output written to: {output_csv}. Processed: {processed}/{len(df)}")


if __name__ == "__main__":
    main()