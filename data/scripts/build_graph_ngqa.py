#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a CSV dataset to per-row JSON graph files with:
- Per-sample node/edge stats
- Row filtering by sample difficulty
- Choice of which difficulty's question/answer to include in the graph
- Start index to skip first N rows
- QUESTION as a separate top-level section (not a node)
- Bidirectional q_ref edges between "question" and the first quoted entity (if present) and 'user' (if present)
- AGENT nodes loaded from an external roles JSON and added as a separate top-level "agents" section
  * Agent entries preserve original fields from roles.json (e.g., top_k, prompt, backbone_id, etc.)
- Agent→Entity linking via OpenAI:
  * Uses OpenAI sync client + ThreadPool (no asyncio) for parallel speed-up
  * Adds bidirectional 'agent_manage' edges between each agent and ALL entities selected by the model
- NEW: Add a top-level 'gold' section (next to 'question') that stores the selected difficulty's gold answer.
- NEW: Save the original CSV 'semantic edge list' as a top-level 'context' (list of [src, rel, tgt]).

Usage:
  python build_graphs_from_csv.py \
    --csv /path/to/data.csv \
    --out_dir /path/to/out_dir \
    --include-difficulties medium \
    --question-difficulty medium \
    --roles-json prompts/roles.json \
    --link-agents \
    --openai-model gpt-4o-mini \
    --max-concurrency 8 \
    --start-index 0 \
    --limit 100

Environment:
  - Set OPENAI_API_KEY in your environment (or pass via --openai-api-key) when using --link-agents.
"""

from __future__ import annotations
import ast
import csv
import json
import argparse
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import concurrent.futures

try:
    # OpenAI Python SDK (>=1.0)
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # We'll error only if --link-agents is used


QUESTION_NODE_NAME = "question"
QUESTION_RELATION = "q_ref"
AGENT_RELATION = "agent_manage"


def _norm(s: str) -> str:
    s2 = (s or "").strip().lower().replace("\u3000", " ")
    s2 = s2.replace("_", " ")
    s2 = " ".join(s2.split())
    return s2


def _norm_diff(s: str) -> str:
    return _norm(s)


def guess_columns(fieldnames: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    norm_map = {_norm(f): f for f in fieldnames}

    def pick(cands: List[str], default: Optional[str] = None) -> Optional[str]:
        for c in cands:
            if c in norm_map:
                return norm_map[c]
        return default

    mapping["difficulty"] = pick(["difficulty", "level"])
    mapping["qe"] = pick(["question easy", "question (easy)", "q easy", "question_easy"])
    mapping["ae"] = pick(["answer easy", "answer (easy)", "a easy", "answer_easy"])
    mapping["qm"] = pick(["question medium", "question (medium)", "q medium", "question_medium"])
    mapping["am"] = pick(["answer medium", "answer (medium)", "a medium", "answer_medium"])
    mapping["qh"] = pick(["question hard", "question (hard)", "q hard", "question_hard"])
    mapping["ah"] = pick(["answer hard", "answer (hard)", "a hard", "answer_hard"])

    mapping["nodes"] = pick(["node list", "nodes", "node_list"])
    mapping["edges"] = pick([
        "sematic edge list",          # common typo
        "semantic edge list",
        "semantic_edge_list",
        "semantic edges",
        "edges semantic",
    ])
    return mapping


def literal_parse(value: str, what: str, row_idx: int) -> Any:
    if value is None:
        raise ValueError(f"Row {row_idx}: {what} is empty/None")
    txt = value.strip()
    if not txt:
        raise ValueError(f"Row {row_idx}: {what} is empty string")
    try:
        return ast.literal_eval(txt)
    except Exception as e:
        preview = txt[:240].replace("\n", " ")
        raise ValueError(f"Row {row_idx}: Failed to parse {what}: {e}; preview={preview!r}")


def extract_nodes(node_list_obj: Any, row_idx: int) -> List[Dict[str, str]]:
    nodes_out: List[Dict[str, str]] = []
    if not isinstance(node_list_obj, (list, tuple)):
        raise ValueError(f"Row {row_idx}: node_list is not a list/tuple")

    for item in node_list_obj:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"Row {row_idx}: node entry has unexpected shape: {item!r}")
        _, meta = item
        if not isinstance(meta, dict):
            raise ValueError(f"Row {row_idx}: node meta is not a dict: {meta!r}")

        node_type_val = meta.get("name", "")
        node_name_val = meta.get("attr", "")

        nodes_out.append({
            "node name": "" if node_name_val is None else str(node_name_val),
            "node type": "" if node_type_val is None else str(node_type_val),
        })
    return nodes_out


def extract_edges(semantic_edges_obj: Any, row_idx: int) -> List[Dict[str, str]]:
    edges_out: List[Dict[str, str]] = []
    if not isinstance(semantic_edges_obj, (list, tuple)):
        raise ValueError(f"Row {row_idx}: semantic edge list is not a list/tuple")

    for triplet in semantic_edges_obj:
        if not (isinstance(triplet, (list, tuple)) and len(triplet) == 3):
            raise ValueError(f"Row {row_idx}: edge triplet has unexpected shape: {triplet!r}")
        src, rel, tgt = triplet
        edges_out.append({
            "source": "" if src is None else str(src),
            "relation": "" if rel is None else str(rel),
            "target": "" if tgt is None else str(tgt),
        })
    return edges_out


def select_qna_from_row(row: Dict[str, str], colmap: Dict[str, str], pick: str) -> Tuple[str, str]:
    pick = _norm_diff(pick)
    if pick not in {"easy", "medium", "hard"}:
        raise ValueError(f"question difficulty must be one of easy|medium|hard, got {pick!r}")
    key_map = {"easy": ("qe", "ae"), "medium": ("qm", "am"), "hard": ("qh", "ah")}
    qkey, akey = key_map[pick]
    qcol = colmap.get(qkey)
    acol = colmap.get(akey)
    q = row.get(qcol, "") if qcol else ""
    a = row.get(acol, "") if acol else ""
    return (q or ""), (a or "")


def _merge_agent_fields(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    protected = {"node name", "node type"}
    for k, v in extra.items():
        if k in {"name", "id", "role", "title", "type"}:
            continue  # already mapped into our canonical fields
        if k not in protected:
            base[k] = v
    return base


def load_agents(roles_path: Path) -> List[Dict[str, Any]]:
    """
    Load agents from roles JSON and convert to dicts:
    {"node name": <agent_name>, "node type": <agent_type or 'agent'>, ...extra fields...}
    Supports list[str], list[dict], dict[str, dict], dict[str, str].
    """
    agents: List[Dict[str, Any]] = []
    if not roles_path.exists():
        return agents

    with roles_path.open("r", encoding="utf-8") as rf:
        try:
            data = json.load(rf)
        except Exception as e:
            raise RuntimeError(f"Failed to parse roles JSON at {roles_path}: {e}")

    def to_agent_from_dict(d: Dict[str, Any], fallback_name: Optional[str] = None) -> Dict[str, Any]:
        name = d.get("name") or d.get("id") or d.get("role") or d.get("title") or fallback_name
        typ  = d.get("type") or d.get("role") or "agent"
        if not name:
            name = json.dumps(d, ensure_ascii=False, sort_keys=True)[:80]
        base = {"node name": str(name), "node type": str(typ)}
        return _merge_agent_fields(base, d)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                agents.append({"node name": item, "node type": "agent"})
            elif isinstance(item, dict):
                agents.append(to_agent_from_dict(item))
            else:
                agents.append({"node name": str(item), "node type": "agent"})
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                agents.append(to_agent_from_dict(v, fallback_name=k))
            elif isinstance(v, str):
                agents.append({"node name": k or v, "node type": "agent"})
            else:
                agents.append({"node name": k, "node type": "agent"})
    else:
        agents.append({"node name": str(data), "node type": "agent"})

    # de-duplicate by ("node name", "node type")
    seen = set()
    out: List[Dict[str, Any]] = []
    for a in agents:
        key = (a.get("node name"), a.get("node type"))
        if key not in seen:
            out.append(a)
            seen.add(key)
    return out


def add_question_links(edges: List[Dict[str, str]], nodes: List[Dict[str, str]], q_text: str) -> None:
    """
    Add bidirectional q_ref edges between 'question' and:
      - the first quoted entity in q_text (if present in nodes)
      - 'user' (if present in nodes)
    """
    if not q_text:
        return
    entity_names = {n.get("node name", "") for n in nodes if isinstance(n, dict)}

    m = re.findall(r'"([^"]+)"', q_text)
    if m:
        ent = m[0]
        if ent in entity_names:
            edges.append({"source": QUESTION_NODE_NAME, "relation": QUESTION_RELATION, "target": ent})
            edges.append({"source": ent, "relation": QUESTION_RELATION, "target": QUESTION_NODE_NAME})

    if "user" in entity_names:
        edges.append({"source": QUESTION_NODE_NAME, "relation": QUESTION_RELATION, "target": "user"})
        edges.append({"source": "user", "relation": QUESTION_RELATION, "target": QUESTION_NODE_NAME})

def add_reverse_entity_edges(edges: List[Dict[str, str]], nodes: List[Dict[str, str]]) -> None:
    """
    为每条 entity-entity 边添加反向边：
      (s, r, t)  ->  (t, r + "_opposite", s)
    仅当 s/t 都是实体名时触发；若 r 已以 "_opposite" 结尾则跳过；已存在则不重复添加。
    """
    # 实体名集合
    entity_names = {n.get("node name", "") for n in nodes if isinstance(n, dict)}
    # 已有边去重表
    existing = set(
        (e.get("source", ""), e.get("relation", ""), e.get("target", ""))
        for e in edges if isinstance(e, dict)
    )
    to_add: List[Dict[str, str]] = []

    for e in list(edges):
        s = e.get("source", "")
        t = e.get("target", "")
        r = e.get("relation", "")
        # 只处理实体-实体；且避免 *_opposite 再加一层
        if (s in entity_names) and (t in entity_names) and isinstance(r, str) and not r.endswith("_opposite"):
            rev = (t, f"{r}_opposite", s)
            if rev not in existing:
                to_add.append({"source": t, "relation": f"{r}_opposite", "target": s})
                existing.add(rev)

    # 统一追加
    edges.extend(to_add)


# ---------------- OpenAI-based agent→entity selection (Threaded) ----------------

def _build_system_prompt(agent: Dict[str, Any]) -> str:
    role = agent.get("role") or agent.get("node type") or "agent"
    backbone = agent.get("backbone_id") or agent.get("backbone") or "<unknown-backbone>"
    name = agent.get("node name", "<agent>")
    return (
        f"You are the agent '{name}'. Role: {role}. Backbone: {backbone}.\n"
        "Mimic the behavior of this specific agent running on the specified backbone.\n"
        "Follow the additional instructions provided next.\n"
        "Think through the task internally, but DO NOT reveal your chain-of-thought; "
        "only output the requested JSON schema."
    )


def _build_behavior_prompt(agent: Dict[str, Any]) -> str:
    p = agent.get("prompt") or agent.get("instructions") or ""
    if not p:
        p = (
            "Behavior: reason carefully, identify the minimal yet complete set of entities "
            "required to justify the final answer according to your strategy."
        )
    return p


def _build_user_prompt(question: str, answer: str, entity_names: List[str], edges: List[Dict[str, str]]) -> str:
    payload = {
        "task": "Select ALL entities that are part of your reasoning chain to reach the answer.",
        "rules": [
            "Reason step-by-step internally, but DO NOT reveal the chain-of-thought.",
            "Your OUTPUT must be valid JSON with the schema below.",
            "Choose only from the provided entity_names exactly as they appear.",
            "Select ALL entities that are meaningfully used along your reasoning path.",
            "Do not include the 'question' pseudo-node. You may include 'user' if used."
        ],
        "question": question or "",
        "answer": answer or "",
        "entity_names": entity_names,
        "graph_edges": edges,
        "output_schema": {"selected_entities": ["<entity_name_1>", "<entity_name_2>", "..."]}
    }
    return json.dumps(payload, ensure_ascii=False)


def _call_openai_sync(
    client: OpenAI,
    agent: Dict[str, Any],
    question: str,
    answer: str,
    entity_names: List[str],
    edges: List[Dict[str, str]],
    model: str,
) -> List[str]:
    sys_prompt = _build_system_prompt(agent)
    behavior = _build_behavior_prompt(agent)
    user_prompt = _build_user_prompt(question, answer, entity_names, edges)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "system", "content": behavior},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},  # force JSON
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        return []

    # Parse JSON
    try:
        data = json.loads(text)
        sel = data.get("selected_entities", [])
        if not isinstance(sel, list):
            return []
        # Keep only exact matches present in entity_names (dedupe preserving order)
        seen = set()
        out: List[str] = []
        for s in sel:
            if isinstance(s, str) and s in entity_names and s not in seen:
                out.append(s)
                seen.add(s)
        return out
    except Exception:
        return []


def link_agents_to_entities(
    graph: Dict[str, Any],
    agents: List[Dict[str, Any]],
    openai_model: str,
    api_key: str,
    timeout: float,          # kept for compatibility; if needed, set via client.with_options(timeout=...)
    max_concurrency: int,
) -> None:
    """
    For each agent, call OpenAI (in parallel via ThreadPool) to select entities,
    then add bidirectional 'agent_manage' edges between the agent and all selected entities.
    """
    if not agents or not api_key or OpenAI is None:
        return

    # Build context
    q_text = ""
    if isinstance(graph.get("question"), list) and graph["question"]:
        q_text = graph["question"][0].get("question", "") or ""
    answer = graph.get("meta", {}).get("answer", "") or ""
    entity_names = [n.get("node name") for n in graph.get("nodes", []) if isinstance(n, dict) and "node name" in n]
    edges_ctx = graph.get("edges", [])

    # Create client (optionally apply timeout)
    client = OpenAI(api_key=api_key)
    if timeout and hasattr(client, "with_options"):
        try:
            client = client.with_options(timeout=timeout)
        except Exception:
            pass  # if not supported, ignore

    # Run in threads
    results: Dict[str, List[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futs = {}
        for agent in agents:
            name = agent.get("node name", "")
            fut = ex.submit(_call_openai_sync, client, agent, q_text, answer, entity_names, edges_ctx, openai_model)
            futs[fut] = name

        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            try:
                results[name] = fut.result() or []
            except Exception:
                results[name] = []

    # Write bidirectional agent_manage edges
    edges = graph.setdefault("edges", [])
    for agent in agents:
        a_name = agent.get("node name")
        if not a_name:
            continue
        for ent in results.get(a_name, []):
            edges.append({"source": a_name, "relation": AGENT_RELATION, "target": ent})
            edges.append({"source": ent, "relation": AGENT_RELATION, "target": a_name})


# ---------------- Graph construction per row ----------------

def build_graph_row(row: Dict[str, str], colmap: Dict[str, str], row_idx: int, pick_qdiff: str, roles_path: Optional[Path], strict: bool = False) -> Dict[str, Any]:
    nodes_raw = row.get(colmap["nodes"]) if colmap.get("nodes") else None
    edges_raw = row.get(colmap["edges"]) if colmap.get("edges") else None

    nodes: List[Dict[str, str]] = []
    edges: List[Dict[str, str]] = []
    context_triplets: List[List[str]] = []  # <--- NEW: preserve original semantic edge list

    if nodes_raw is None:
        msg = f"Row {row_idx}: nodes column not found or empty"
        if strict:
            raise ValueError(msg)
    else:
        nodes = extract_nodes(literal_parse(nodes_raw, "node_list", row_idx), row_idx)

    if edges_raw is None:
        msg = f"Row {row_idx}: semantic edges column not found or empty"
        if strict:
            raise ValueError(msg)
    else:
        # Parse once → keep as 'context' AND convert to dict edges for graph construction
        sem_edges_obj = literal_parse(edges_raw, "semantic edge list", row_idx)
        # Keep original form as `context` (ensure it's a list of triplets)
        if isinstance(sem_edges_obj, (list, tuple)):
            context_triplets = []
            for triplet in sem_edges_obj:
                if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                    s, r, t = triplet
                    context_triplets.append([("" if s is None else str(s)),
                                             ("" if r is None else str(r)),
                                             ("" if t is None else str(t))])
        # Build normalized dict edges for the graph
        edges = extract_edges(sem_edges_obj, row_idx)

    # Select the requested difficulty's Q/A
    q_txt, a_txt = select_qna_from_row(row, colmap, pick_qdiff)
    difficulty_val = row.get(colmap["difficulty"], "") if colmap.get("difficulty") else ""

    # Build question top-level section (not in nodes)
    question_section: List[Dict[str, str]] = []
    if q_txt:
        question_section.append({"node name": QUESTION_NODE_NAME, "question": q_txt})
        # Add q_ref links (bidirectional) to first quoted entity and to user (if present)
        add_question_links(edges, nodes, q_txt)

        # 为实体-实体边补反向边（关系名加 "_opposite"）
        add_reverse_entity_edges(edges, nodes)


    # # NEW: Build gold top-level section with the selected difficulty's gold answer
    # gold_section: List[Dict[str, str]] = []
    # if a_txt:
    #     gold_section.append({"node name": "gold", "answer": a_txt})

    # Load agent nodes (preserve extra fields; e.g., top_k, prompt, backbone_id)
    agents: List[Dict[str, Any]] = []
    if roles_path is not None:
        agents = load_agents(roles_path)

    meta: Dict[str, Any] = {
        "difficulty": difficulty_val,
        "question": q_txt,
        "answer": a_txt,
    }

    # Keep key order: meta -> question -> context -> nodes -> agents -> edges
    return {
        "meta": meta,
        "question": question_section,
        "context": context_triplets,  # <--- NEW
        "nodes": nodes,
        "agents": agents,
        "edges": edges
    }


def parse_include_difficulties(text: str) -> Set[str]:
    items = [t.strip() for t in (text or "").split(",") if t.strip()]
    if not items:
        return {"medium"}  # default
    out = set()
    for it in items:
        itn = _norm_diff(it)
        if itn not in {"easy", "medium", "hard"}:
            raise ValueError(f"Invalid difficulty in --include-difficulties: {it!r}")
        out.add(itn)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV rows into JSON graph files with difficulty filtering, selectable question difficulty, start index, top-level question, gold, q_ref links, agents, and threaded OpenAI-based agent linking."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--out_dir", required=True, help="Directory to write JSON files")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows (0 = all)")
    parser.add_argument("--encoding", default="utf-8", help="CSV file encoding (default: utf-8)")
    parser.add_argument("--strict", action="store_true", help="Fail on parse errors instead of skipping row")

    # Column overrides
    parser.add_argument("--col-nodes", dest="col_nodes", default=None, help="Column name for node_list")
    parser.add_argument("--col-edges", dest="col_edges", default=None, help="Column name for semantic edge list")
    parser.add_argument("--col-difficulty", dest="col_diff", default=None)
    parser.add_argument("--col-qe", dest="col_qe", default=None)
    parser.add_argument("--col-ae", dest="col_ae", default=None)
    parser.add_argument("--col-qm", dest="col_qm", default=None)
    parser.add_argument("--col-am", dest="col_am", default=None)
    parser.add_argument("--col-qh", dest="col_qh", default=None)
    parser.add_argument("--col-ah", dest="col_ah", default=None)

    # Controls
    parser.add_argument("--include-difficulties", default="medium",
                        help="Comma-separated difficulties to include (subset of: easy,medium,hard). Default: medium")
    parser.add_argument("--question-difficulty", default="medium",
                        help="Which difficulty's question/answer to put into meta (one of: easy,medium,hard). Default: medium")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N data rows (0-based). Example: --start-index 100 skips the first 100 rows.")
    parser.add_argument("--roles-json", default="prompts/roles.json",
                        help="Path to roles JSON file for agent nodes (default: prompts/roles.json). If not found, agents list will be empty.")

    # OpenAI linking controls (threaded)
    parser.add_argument("--link-agents", action="store_true",
                        help="Enable OpenAI-based linking from agents to entities via bidirectional 'agent_manage' edges.")
    parser.add_argument("--openai-model", default="gpt-4o-mini",
                        help="OpenAI model to use for agent linking (default: gpt-4o-mini).")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""),
                        help="OpenAI API key. If empty, will read from OPENAI_API_KEY env var.")
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="Max concurrent OpenAI requests per row (default: 8).")
    parser.add_argument("--request-timeout", type=float, default=60.0,
                        help="Per-request timeout in seconds (best-effort via client.with_options).")

    args = parser.parse_args()

    in_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    include_set = parse_include_difficulties(args.include_difficulties)
    pick_qdiff = _norm_diff(args.question_difficulty)
    if pick_qdiff not in {"easy", "medium", "hard"}:
        raise ValueError("--question-difficulty must be one of: easy, medium, hard")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    roles_path = Path(args.roles_json) if args.roles_json else None

    with in_path.open("r", encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError("CSV appears to have no header row.")

        colmap = guess_columns(fieldnames)
        if args.col_nodes: colmap["nodes"] = args.col_nodes
        if args.col_edges: colmap["edges"] = args.col_edges
        if args.col_diff:  colmap["difficulty"] = args.col_diff
        if args.col_qe:    colmap["qe"] = args.col_qe
        if args.col_ae:    colmap["ae"] = args.col_ae
        if args.col_qm:    colmap["qm"] = args.col_qm
        if args.col_am:    colmap["am"] = args.col_am
        if args.col_qh:    colmap["qh"] = args.col_qh
        if args.col_ah:    colmap["ah"] = args.col_ah

        processed = 0
        total_nodes = 0
        total_edges = 0
        row_idx = 1  # 1-based data row index
        for row in reader:

            # Skip the first N data rows based on --start-index (0 means start from the first row)
            if row_idx <= args.start_index:
                continue

            if args.limit and processed >= args.limit:
                break

            # Filter by difficulty
            diff_raw = row.get(colmap["difficulty"], "") if colmap.get("difficulty") else ""
            diff_norm = _norm_diff(diff_raw)
            if diff_norm and include_set and (diff_norm not in include_set):
                continue

            try:
                graph = build_graph_row(row, colmap, row_idx, pick_qdiff=pick_qdiff, roles_path=roles_path, strict=args.strict)
            except Exception as e:
                print(f"[WARNING] Skipping row {row_idx} due to error: {e}")
                if args.strict:
                    raise
                continue

            # Optionally link agents to entities via OpenAI (threaded)
            agents = graph.get("agents", [])
            if args.link_agents and agents and args.openai_api_key:
                try:
                    link_agents_to_entities(
                        graph=graph,
                        agents=agents,
                        openai_model=args.openai_model,
                        api_key=args.openai_api_key,
                        timeout=args.request_timeout,
                        max_concurrency=args.max_concurrency,
                    )
                except Exception as e:
                    print(f"[WARNING] Agent linking failed on row {row_idx}: {e}")

            n_nodes = len(graph["nodes"])
            n_edges = len(graph["edges"])
            total_nodes += n_nodes
            total_edges += n_edges
            fname = f"graph_{row_idx:06d}.json"
            out_file = out_dir / fname
            with out_file.open("w", encoding="utf-8") as wf:
                json.dump(graph, wf, ensure_ascii=False, indent=2)

            processed += 1
            print(f"[row {row_idx} | diff={diff_norm or 'NA'}] nodes={n_nodes} edges={n_edges} -> {fname} (Q/A from: {pick_qdiff})")
            row_idx += 1

        print(f"✅ Done. Wrote {processed} JSON files to: {out_dir}")
        print(f"📊 Total nodes across processed rows: {total_nodes}")
        print(f"📈 Total edges across processed rows: {total_edges}")


if __name__ == "__main__":
    main()
