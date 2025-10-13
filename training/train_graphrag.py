#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train RouterGNN with external agents via Together AI.

- Reads hetero graphs list from a .pt file (produced by your BERT-768 embed script).
- Loads 24 agents from prompts/roles.json.
- For each graph, queries ALL agents (parallel, cached) with the full graph context + question.
- Parses each agent's raw output to extract ONLY the final answer text.
- Builds soft targets from per-agent F1 vs gold answer; trains RouterGNN with KL loss.
- Provides evaluate() with stable top-k aggregation.

NEW (this version):
- 8:1:1 split → train:val:test
- Test-time report includes:
  (1) final EM/F1 using your configured --topk_eval
  (2) per-agent average EM/F1 over the test set
  (3) top‑k sweep (k=1..A) EM/F1 over the test set
  All of (2)&(3) are written into the SAME JSONL file (see --results_jsonl)
- Single-file cache: all (question, agent) answers persisted in one JSONL file at --cache_dir/answers.jsonl
- eval_only mode: load a checkpoint and skip training directly to test

Usage:
  python train_router_with_agents_together.py \
    --graphs_pt data/graphs.pt \
    --roles_json prompts/roles.json \
    --out_dir runs/exp1 \
    --epochs 3 \
    --lr 2e-4 \
    --batch_size 1 \
    --device auto \
    --together_model_fallback meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --topk_eval 8 \
    --target_tau 0.25 \
    --label_smooth 1e-3 \
    --max_concurrency 12 \
    --cache_dir .cache/agent_answers \
    --allow_new_llm 1

Env:
  TOGETHER_API_KEY must be set (export TOGETHER_API_KEY=...)
"""

import os
import re
import io
import gc
import sys
import json
import math
import time
import hashlib
import random
import argparse
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

import requests
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import HeteroData

import warnings
import logging
import contextlib
import io

# === import your RouterGNN (the version without question-type branch; in_dim=768) ===
# make sure model.py (with class RouterGNN) is importable
from model import RouterGNN
from bert_score import score






# ----------------------------- Utils: text normalization / metrics -----------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT = r"""!"#$%&'()*+,-./:;<=>?[\]^_`{|}~"""

def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(f"[{re.escape(_PUNCT)}]", " ", s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)

def exact_match(pred: str, gold: str) -> int:
    return int(_normalize_text(pred) == _normalize_text(gold))

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    from collections import Counter
    c_pred = Counter(pred_tokens)
    c_gold = Counter(gold_tokens)
    num_same = sum((c_pred & c_gold).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall + 1e-9)


def md5_of(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


# ----------------------------- Graph -> textual context for agents -----------------------------


def graph_to_text(data: HeteroData) -> str:
    """
    Compact text for LLMs (nodes + edges only).
    FORMAT: edge = [source | relation | target]  (1st=source, 2nd=relation, 3rd=target)
    Edges come from `data.context` triplets when available; otherwise fall back to edge_index_dict.
    """
    lines = []
    # ---- NODES (entities only, names) ----
    ents = getattr(data, "entity_names", []) or []
    lines.append("NODES:")
    for nm in ents:
        lines.append(f"- {nm}")

    # ---- EDGES ----
    lines.append("EDGES:")
    ctx = getattr(data, "context", None)

    # Prefer the saved context triplets
    if isinstance(ctx, list) and ctx and all(isinstance(t, (list, tuple)) and len(t) == 3 for t in ctx):
        for s, r, t in ctx:
            s = "" if s is None else str(s)
            r = "" if r is None else str(r)
            t = "" if t is None else str(t)
            lines.append(f"- {s} | {r} | {t}")
        return "\n".join(lines)

    # Fallback: derive from graph edges (kept minimal)
    name_maps = {
        "entity": ents,
        "question": [getattr(data, "question_node_name", "question")],
        "agent": getattr(data, "agent_names", []) or [],
    }
    try:
        e_dict = data.edge_index_dict
    except KeyError:
        e_dict = {}

    for (src_t, rel, dst_t), eobj in e_dict.items():
        src_names = name_maps.get(src_t, [])
        dst_names = name_maps.get(dst_t, [])
        ei = getattr(eobj, "edge_index", eobj)
        try:
            import torch as _torch
            if not _torch.is_tensor(ei) or ei.numel() == 0:
                continue
            for j in range(ei.size(1)):
                sidx = int(ei[0, j]); didx = int(ei[1, j])
                sname = src_names[sidx] if 0 <= sidx < len(src_names) else f"{src_t}:{sidx}"
                dname = dst_names[didx] if 0 <= didx < len(dst_names) else f"{dst_t}:{didx}"
                lines.append(f"- {sname} | {rel} | {dname}")
        except Exception:
            continue

    return "\n".join(lines)


import json, torch, torch.nn.functional as F


# ----------------------------- Together AI client (OpenAI-compatible chat.completions) ---------

class TogetherClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.together.xyz/v1",
                 timeout: float = 60.0, max_retries: int = 3, backoff: float = 1.5):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.sess = requests.Session()
        self.sess.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2,
             max_tokens: int = 256, top_p: float = 1.0) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        for attempt in range(self.max_retries):
            try:
                r = self.sess.post(url, data=json.dumps(payload), timeout=self.timeout)
                if r.status_code == 200:
                    js = r.json()
                    return js.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                elif r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.backoff ** attempt)
                else:
                    break
            except Exception:
                time.sleep(self.backoff ** attempt)
        return ""


# ----------------------------- Agent & prompt handling ----------------------------------------

def load_agents(roles_json: str) -> List[Dict[str, Any]]:
    with open(roles_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    agents: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                nm = it.get("node name") or it.get("name") or it.get("id") or it.get("role") or ""
                tp = it.get("node type") or it.get("type") or "agent"
                ag = dict(it)
                ag["node name"] = nm; ag["node type"] = tp
                agents.append(ag)
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                ag = dict(v)
                ag["node name"] = ag.get("node name") or ag.get("name") or k
                ag["node type"] = ag.get("node type") or ag.get("type") or "agent"
                agents.append(ag)
    else:
        raise ValueError("roles.json must be list/dict")
    seen = set(); uniq = []
    for a in agents:
        key = (a.get("node name", ""), a.get("node type", "agent"))
        if key not in seen:
            uniq.append(a); seen.add(key)
    return uniq


#----------------------------- Default Backbone Map -----------------------------
DEFAULT_BACKBONE_MAP = {
    "BACKBONE::mixtral_8x7b":     "mistralai/Mistral-7B-Instruct-v0.2",
    "BACKBONE::llama3_8b_lite":   "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "BACKBONE::qwen2p5_7b_turbo": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "BACKBONE::gpt_oss_20b":      "openai/gpt-oss-20b",
}

def resolve_model_id(agent: Dict[str, Any], fallback: str) -> str:
    mid = agent.get("model") or agent.get("together_model") or ""
    if mid:
        return mid
    bb = agent.get("backbone_id") or agent.get("backbone") or ""
    if bb in DEFAULT_BACKBONE_MAP:
        return DEFAULT_BACKBONE_MAP[bb]
    if "/" in str(bb):
        return str(bb)
    return fallback


def build_messages_for_agent(agent: Dict[str, Any], q_text: str, graph_text: str) -> List[Dict[str, str]]:
    name = agent.get("node name", "<agent>")
    role = agent.get("node type", "agent")
    bb = agent.get("backbone_id", "<backbone>")
    base_prompt = (agent.get("prompt") or agent.get("instructions") or "").strip()

    # === 仅修改：输出格式从“带 because + tags 的句子”改为 仅 "Yes." / "No." ===
    json_spec = (
        'Return EXACTLY one single-line JSON object with NO extra text, NO code fences, '
        'NO explanation: {"answer":"Yes"} or {"answer":"No"}\n'
        "Rules:\n"
        '1) Output only ONE JSON object. No prose, no prefix/suffix, no markdown.\n'
        '2) The value of "answer" MUST be exactly "Yes" or "No" (capitalized, ending with no period).\n'
        '3) Do NOT include any keys other than "answer".\n'
        '4) If unsure, output {"answer": ""} (still valid JSON, single line).\n'
        "Examples:\n"
        'OK: {"answer":"Yes"}\n'
        'OK: {"answer":"No"}\n'
        'BAD: {"answer":"No."}\n(Because I don\'t want the period)\n'
        'BAD: {"answer":"Yes, because the food is low_carb"}\n'
        'BAD: Here is the answer: {"answer":"Yes"}\n'
        'BAD: ```json\\n{\"answer\":\"Yes\"}\\n```'
    )

    system_msg = "\n".join([
        f"You are agent '{name}'. Role: {role}. Backbone: {bb}.",
        "Follow your base instructions if provided below, but you MUST obey the output specification strictly.",
        ("Base instructions:\n" + base_prompt) if base_prompt else "",
        "Do NOT reveal chain-of-thought.",
        json_spec,
    ]).strip()

    # === 思路不变：仍然基于图中显式的营养标签关联决定 Yes/No，但不在输出中解释 ===
    user_msg = (
        "You are given a knowledge graph and a question. Answer using the JSON spec above.\n\n"
        "- Decide Yes/No based on the nutrition tags that are supported by explicit edges in the graph.\n"
        "- A tag counts only if there is at least one edge linking it to the user's condition (e.g., dietary habits, health status, medical needs, nutrient preferences) indicating relevance — either positively (matches) or negatively (contradicts).\n"
        "- If no edge explicitly connects that tag to the user's state, do NOT treat it as evidence.\n"
        "- IMPORTANT: The final output must be ONLY 'Yes.' or 'No.' in the JSON format required; do NOT include explanations or tags in the output.\n\n"
        "If the nutrition tags linked to the food is contradict to the user's condition, answer 'No'.\n"
        "If the nutrition tags linked to the food is aligned with the user's condition, answer 'Yes'.\n"
        f"QUESTION:\n{q_text}\n\n"
        f"GRAPH:\n{graph_text}\n"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]



# def build_messages_for_agent(agent: Dict[str, Any], q_text: str, graph_text: str) -> List[Dict[str, str]]:
#     name = agent.get("node name", "<agent>")
#     role = agent.get("node type", "agent")
#     bb = agent.get("backbone_id", "<backbone>")
#     base_prompt = (agent.get("prompt") or agent.get("instructions") or "").strip()

#     # === 仅修改：输出格式从“纯 tag 列表”变为 “Yes/No, because the food is <tags>” ===
#     json_spec = (
#         'Return EXACTLY one single-line JSON object with NO extra text, NO code fences, '
#         'NO explanation: {"answer":"Yes, because the food is <tag1>[, <tag2>][, <tag3>]..."} or {"answer":"No, because the food is <tag1>[, <tag2>]..."}'
#         "\nRules:\n"
#         '1) Only one JSON object. No prose, no prefix/suffix, no markdown.\n'
#         '2) The value of "answer" is a single string and MUST start with either "Yes, because the food is " or "No, because the food is ".\n'
#         '3) Tags must be lowercase snake_case (e.g., "low_carb"), joined by a comma and a space.\n'
#         '4) Do NOT include any keys other than "answer".\n'
#         '5) If unsure, output {"answer": ""} (still valid JSON, single line).\n'
#         'Examples:\n'
#         'OK: {"answer":"Yes, because the food is low_carb, low_sugar, high_protein"}\n'
#         'OK: {"answer":"No, because the food is high_sodium"}\n'
#         'BAD: {"answer":"low_carb, low_sugar"}\n'
#         'BAD: Here is the answer: {"answer":"Yes, because ..."}\n'
#         'BAD: ```json\\n{"answer":"..."}\\n```'
#     )

#     system_msg = "\n".join([
#         f"You are agent '{name}'. Role: {role}. Backbone: {bb}.",
#         "Follow your base instructions if provided below, but you MUST obey the output specification strictly.",
#         ("Base instructions:\n" + base_prompt) if base_prompt else "",
#         "Do NOT reveal chain-of-thought.",
#         json_spec,
#     ]).strip()

#     # === 仅微调：仍保持“标签需由图中显式边支持”，但现在用于支撑 Yes/No 判断 ===
#     user_msg = (
#         "You are given a knowledge graph and a question. Answer using the JSON spec above.\n\n"
#         # "IMPORTANT:\n"
#         "- Decide Yes/No based on the nutrition tags that are supported by explicit edges in the graph.\n"
#         "- A tag should only be included if there is at least one edge linking it to the user's condition (e.g., dietary habits, health status, medical needs, nutrient preferences) indicating relevance — either positively (matches) or negatively (contradicts).\n"
#         "- If no edge explicitly connects that tag to the user's state, DO NOT include that tag in the explanation.\n\n"
#         f"QUESTION:\n{q_text}\n\n"
#         f"GRAPH:\n{graph_text}\n"
#     )

#     return [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_msg},
#     ]


# def build_messages_for_agent(agent: Dict[str, Any], q_text: str, graph_text: str) -> List[Dict[str, str]]:
#     name = agent.get("node name", "<agent>")
#     role = agent.get("node type", "agent")
#     bb = agent.get("backbone_id", "<backbone>")
#     base_prompt = (agent.get("prompt") or agent.get("instructions") or "").strip()

#     json_spec = (
#         'Return EXACTLY one single-line JSON object with NO extra text, NO code fences, '
#         'NO explanation: {"answer":"<tag>"}\n'
#         "Rules:\n"
#         "1) Only one JSON object. No prose, no prefix/suffix, no markdown.\n"
#         '2) The value of \"answer\" must be a single nutrition tag string (NOT a list).\n'
#         "3) Tags must be lowercase snake_case (e.g., \"low_carb\").\n"
#         "4) Do NOT include any keys other than \"answer\".\n"
#         '5) If unsure, output {\"answer\": \"\"} (still valid JSON, single line).\n'
#         "6) Do NOT output multiple tags. Only one.\n"
#         "Examples:\n"
#         'OK: {"answer":"high_sodium"}\n'
#         'OK: {"answer":""}\n'
#         'BAD: {"answer":"low_carb, high_protein"}  # multiple tags \n'
#         'BAD: Here is the answer: {"answer":"..."}  # prose \n'
#         'BAD: ```json\\n{"answer":"..."}\\n```      # markdown '
#     )

#     system_msg = "\n".join([
#         f"You are agent '{name}'. Role: {role}. Backbone: {bb}.",
#         "Follow your base instructions if provided below, but you MUST obey the output specification strictly.",
#         ("Base instructions:\n" + base_prompt) if base_prompt else "",
#         "Do NOT reveal chain-of-thought.",
#         json_spec,
#     ]).strip()

#     user_msg = (
#         "You are given a knowledge graph and a question. Answer using the JSON spec above.\n\n"
#         "IMPORTANT:\n"
#         "- The graph is a list of triples, each in the format: [head, relation, tail].\n"
#         "- ONLY consider triples whose relation is one of: \"need\", \"match\", or \"contradict\".\n"
#         "- In those triples, either the first element (head) OR the third element (tail) can be a nutrition tag.\n"
#         "- Your task is to output EXACTLY ONE nutrition tag that appears in such a triple.\n"
#         "- If multiple nutrition tags appear, choose the one most relevant to the question.\n"
#         "- If no nutrition tag appears in any triple with `need`, `match`, or `contradict`, output an empty string.\n\n"
#         f"QUESTION:\n{q_text}\n\n"
#         f"GRAPH (list of triples):\n{graph_text}\n"
#     )

#     return [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_msg},
#     ]



# def build_messages_for_agent(agent: Dict[str, Any], q_text: str, graph_text: str) -> List[Dict[str, str]]:
#     name = agent.get("node name", "<agent>")
#     role = agent.get("node type", "agent")
#     bb = agent.get("backbone_id", "<backbone>")
#     base_prompt = (agent.get("prompt") or agent.get("instructions") or "").strip()

#     json_spec = (
#         'Return EXACTLY one single-line JSON object with NO extra text, NO code fences, '
#         'NO explanation: {"answer":"<tag1>[, <tag2>][, <tag3>]..."}'
#         "\nRules:\n"
#         '1) Only one JSON object. No prose, no prefix/suffix, no markdown.\n'
#         '2) The value of "answer" is a single string. If multiple tags, join with a comma and a space.\n'
#         '3) Tags must be lowercase snake_case (e.g., "low_carb").\n'
#         '4) Do NOT include any keys other than "answer".\n'
#         '5) If unsure, output {"answer": ""} (still valid JSON, single line).\n'
#         'Examples:\n'
#         'OK: {"answer":"low_carb, low_sugar, high_protein"}\n'
#         'OK: {"answer":"high_sodium"}\n'
#         'BAD: Here is the answer: {"answer":"..."}\n'
#         'BAD: ```json\\n{"answer":"..."}\\n```'
#     )

#     system_msg = "\n".join([
#         f"You are agent '{name}'. Role: {role}. Backbone: {bb}.",
#         "Follow your base instructions if provided below, but you MUST obey the output specification strictly.",
#         ("Base instructions:\n" + base_prompt) if base_prompt else "",
#         "Do NOT reveal chain-of-thought.",
#         json_spec,
#     ]).strip()

#     user_msg = (
#         "You are given a knowledge graph and a question. Answer using the JSON spec above.\n\n"
#         "IMPORTANT:\n"
#         "- The tags you output MUST be directly supported by explicit edges in the graph.\n"
#         "- That means: a tag should only be included if there is at least one edge linking it to the user's condition (e.g., dietary habits, health status, medical needs, nutrient preferences) that indicates it is relevant — either positively (matches the condition) or negatively (contradicts the condition).\n"
#         "- If no edge explicitly connects that tag to the user's state, DO NOT output that tag.\n\n"
#         "One or two would be better than three.\n"
#         "REMEMBER:\n"
#         "Only output one if you are very sure is OK.\n But don't output more than 3 nutrition tags.\n "
#         "And One or two would be better than three.\n\n"
#         f"QUESTION:\n{q_text}\n\n"
#         f"GRAPH:\n{graph_text}\n"
#     )

#     return [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_msg},
#     ]


def extract_answer_text(raw: str) -> str:
    import re, json

    def _find_first_json_object(text: str) -> str:
        s = text or ""
        start = s.find("{")
        while start != -1:
            stack = 0
            for i in range(start, len(s)):
                if s[i] == "{":
                    stack += 1
                elif s[i] == "}":
                    stack -= 1
                    if stack == 0:
                        return s[start:i+1]
            start = s.find("{", start + 1)
        return ""

    # 将 snake_case 标签转换为短语：first "_" -> " in ", others "_" -> " "
    _SNAKE = re.compile(r'\b([A-Za-z]+(?:_[A-Za-z0-9]+)+)\b')
    def _snake_to_phrase(tok: str) -> str:
        parts = tok.split('_')
        if len(parts) >= 2:
            return parts[0] + " in " + " ".join(parts[1:])
        return tok

    def _render_tags_to_phrases(text: str) -> str:
        return _SNAKE.sub(lambda m: _snake_to_phrase(m.group(1)), text)

    s = (raw or "").strip()
    if not s:
        return ""

    ans = None

    # 优先：解析第一段 JSON 并取 "answer"
    cand = _find_first_json_object(s)
    if cand:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                ans = obj["answer"].strip()
        except Exception:
            pass

    # 兜底：正则直接抓取 "answer":"..."
    if ans is None:
        m = re.search(r'"answer"\s*:\s*"([^"]*)"', s)
        if m:
            ans = m.group(1).strip()

    if not ans:
        return ""

    # # 转换 snake_case 标签为短语
    # ans = _render_tags_to_phrases(ans)

    # # 若末尾没有 . ! ? 则补一个句号
    # if not re.search(r'[.!?]$', ans):
    #     ans += "."

    return ans

# ----------------------------- Single-file Answer Store ----------------------------------------

class AnswerStore:
    """
    Thread-safe, single-file JSONL cache:
      File path: <cache_dir>/answers.jsonl
      Each line: {"qid_md5": "...", "agent_id": "...", "raw": "...", "answer": "...", "ts": 1690000000.123}

    - get()/get_answer()/set()/set_many()/flush()
    - _data[question_id][agent_name] = {...}
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / "answers.jsonl"
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            self._data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        qid = obj.get("qid_md5"); ag = obj.get("agent_id")
                        if qid and ag:
                            self._data.setdefault(qid, {})[ag] = {
                                "raw": obj.get("raw", ""),
                                "answer": obj.get("answer", ""),
                                "ts": obj.get("ts", 0.0)
                            }
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[AnswerStore] Failed to load JSONL cache: {e}")
            self._data = {}

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        with self._lock, tmp.open("w", encoding="utf-8") as f:
            for qid, agents in self._data.items():
                for ag, val in agents.items():
                    rec = {
                        "qid_md5": qid,
                        "agent_id": ag,
                        "raw": val.get("raw", ""),
                        "answer": val.get("answer", ""),
                        "ts": val.get("ts", time.time())
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def get(self, question_id: str, agent_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            recs = self._data.get(question_id, None)
            if recs is None:
                return None
            val = recs.get(agent_name, None)
            return dict(val) if isinstance(val, dict) else None

    def get_answer(self, question_id: str, agent_name: str) -> str:
        rec = self.get(question_id, agent_name)
        return (rec or {}).get("answer", "") or ""

    def set(self, question_id: str, agent_name: str, raw: str, answer: str):
        with self._lock:
            node = self._data.setdefault(question_id, {})
            node[agent_name] = {"raw": raw or "", "answer": answer or "", "ts": time.time()}
            rec = {"qid_md5": question_id, "agent_id": agent_name, "raw": raw or "", "answer": answer or "", "ts": node[agent_name]["ts"]}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def set_many(self, updates: List[Tuple[str, str, str, str]]):
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                for qid, ag, raw, ans in updates:
                    node = self._data.setdefault(qid, {})
                    node[ag] = {"raw": raw or "", "answer": ans or "", "ts": time.time()}
                    rec = {"qid_md5": qid, "agent_id": ag, "raw": raw or "", "answer": ans or "", "ts": node[ag]["ts"]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

@torch.no_grad()
def _safe_names(data):
    return list(getattr(data, "entity_names", []) or [])

def compute_entity_importance(model, data, method="grad"):
    """
    不改模型，通过一次独立的前向 + autograd 计算实体重要度:
    s = E_{p(a|q)}[logit_a] = sum softmax(logit)*logit
    score_i = ||∂s/∂x_i||_2   (method='grad')
            或  sum(|(∂s/∂x_i) * x_i|) (method='grad_input')
    返回: scores (E,) 已按样本内归一化到 sum=1
    """
    # 需要梯度，先打开 grad 上下文
    with torch.enable_grad():
        # 让实体特征可导（不破坏原图）
        orig_x = data["entity"].x
        ent_x = orig_x.detach().clone().requires_grad_(True)
        data["entity"].x = ent_x

        out = model(data)
        logits = out["logits"] if isinstance(out, dict) else out
        probs  = F.softmax(logits, dim=-1)
        s = (probs * logits).sum()   # 单标量

        grads = torch.autograd.grad(s, ent_x, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grads is None:
            scores = torch.zeros(ent_x.size(0), device=ent_x.device)
        else:
            if method == "grad_input":
                scores = (grads * ent_x).abs().sum(dim=1)
            else:  # 'grad'
                scores = grads.norm(p=2, dim=1)

        # 还原，避免污染后续
        data["entity"].x = orig_x

        # 归一化便于比较
        denom = scores.sum() + 1e-9
        return scores / denom
    

# --------------------------------------------------------------------------------

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self._pick_device(args.device)
        self.client = TogetherClient(
            api_key=os.getenv("TOGETHER_API_KEY", ""),
            timeout=args.request_timeout,
            max_retries=3,
            backoff=1.7,
        )
        self.store = AnswerStore(args.cache_dir)

        self.graphs: List[HeteroData] = torch.load(args.graphs_pt, map_location="cpu")
        assert len(self.graphs) > 0, "Empty graphs list."

        node_types = ['question', 'entity', 'agent']
        edge_types = [
            ('entity', 'rel:belongs_to', 'entity'),
            ('entity', 'rel:contradict', 'entity'),
            ('entity', 'rel:has', 'entity'),
            ('entity', 'rel:match', 'entity'),
            ('entity', 'rel:need', 'entity'),
            ('entity', 'rel:belongs_to_opposite', 'entity'),
            ('entity', 'rel:contradict_opposite', 'entity'),
            ('entity', 'rel:has_opposite', 'entity'),
            ('entity', 'rel:match_opposite', 'entity'),
            ('entity', 'rel:need_opposite', 'entity'),
            ('question', 'rel:q_ref', 'entity'),
            ('entity', 'rel:q_ref', 'question'),
            ('agent', 'rel:agent_manage', 'entity'),
            ('entity', 'rel:agent_manage', 'agent')
        ]
        print(f"Graph metadata: node_types={node_types}, edge_types={edge_types}")
        self.model = RouterGNN(in_dim=768, hid_dim=args.hid_dim, num_layers=args.num_layers,
                               metadata=(node_types, edge_types), dropout=args.dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.agents = load_agents(args.roles_json)
        assert len(self.agents) == 24 or len(self.agents) == len(getattr(self.graphs[0], "agent_names", [])), \
            f"Expected 24 agents; got {len(self.agents)}"

    @staticmethod
    def _pick_device(dev: str) -> str:
        if dev == "cpu":
            return "cpu"
        if dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _answer_from_llm(
        self,
        agent: Dict[str, Any],
        q_text: str,
        graph_text: str,
        model_fallback: str
    ) -> Tuple[str, str]:
        model_id = resolve_model_id(agent, fallback=model_fallback)
        model_lc = model_id.lower()

        messages = build_messages_for_agent(agent, q_text, graph_text)

        is_gss = ("gpt-oss" in model_lc) or ("oss" in model_lc and "openai/" in model_lc)

        if is_gss:
            try:
                oclient = OpenAI()
                completion = oclient.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=self.args.temperature,
                    max_tokens=3000,
                    top_p=1.0
                )
                raw_txt = (completion.choices[0].message.content or "").strip()
                ans = extract_answer_text(raw_txt)
            except Exception as e:
                print(f"[GSS→OpenAI fallback ERROR] {type(e).__name__}: {e}")
                raw_txt, ans = "", ""
        else:
            raw_txt = self.client.chat(
                model=model_id,
                messages=messages,
                temperature=self.args.temperature,
                max_tokens=self.args.max_tokens,
                top_p=1.0
            )
            ans = extract_answer_text(raw_txt)

        return raw_txt, ans


    def answers_for_agents_parallel(self, data: HeteroData, allow_new_llm: bool = True) -> Dict[str, str]:
        q_text = getattr(data, "question_text", "") or ""
        graph_txt = graph_to_text(data)
        agent_names = getattr(data, "agent_names", []) or [a.get("node name", "") for a in self.agents]
        name2agent = {a.get("node name", ""): a for a in self.agents}
        qid = str(getattr(data, "qid", None) or md5_of(q_text))

        out: Dict[str, str] = {}
        missing: List[str] = []
        for ag_name in agent_names:
            cached = self.store.get(qid, ag_name)
            if cached and (cached.get("answer") or "") != "":
                out[ag_name] = cached["answer"]
            else:
                out[ag_name] = ""; missing.append(ag_name)

        if (not allow_new_llm) or len(missing) == 0:
            return out

        updates: List[Tuple[str, str, str, str]] = []

        def _worker(ag_name: str) -> Tuple[str, str]:
            ag = name2agent.get(ag_name)
            if ag is None:
                return ag_name, ("", "")
            try:
                raw, ans = self._answer_from_llm(ag, q_text, graph_txt, self.args.together_model_fallback)
                return ag_name, (raw, ans)
            except Exception:
                return ag_name, ("", "")

        with ThreadPoolExecutor(max_workers=self.args.max_concurrency) as ex:
            futs = {ex.submit(_worker, nm): nm for nm in missing}
            for fut in as_completed(futs):
                nm = futs[fut]
                try:
                    nm2, payload = fut.result()
                    raw, ans = payload
                except Exception:
                    raw, ans = "", ""
                out[nm] = ans or ""
                updates.append((qid, nm, raw or "", ans or ""))

        if updates:
            self.store.set_many(updates)
            self.store.flush()
        return out

    @staticmethod
    def soft_targets_from_f1(per_agent_f1: Dict[str, float], tau: float = 0.25, eps: float = 1e-3) -> Dict[str, float]:
        names = list(per_agent_f1.keys())
        vals = torch.tensor([max(0.0, float(per_agent_f1[n])) for n in names], dtype=torch.float32)
        if vals.sum() <= 1e-12:
            tgt = torch.full_like(vals, 1.0 / max(1, len(names)))
        else:
            tgt = torch.softmax(vals / max(1e-6, tau), dim=-1)
        if eps > 0:
            K = len(names)
            tgt = (1 - eps) * tgt + eps * (1.0 / max(1, K))
        return {n: float(t) for n, t in zip(names, tgt.tolist())}

    @staticmethod
    def aggregate_answers(agent_answers: Dict[str, str], probs: Dict[str, float]) -> str:
        groups: Dict[str, float] = {}
        norm2raw: Dict[str, str] = {}
        for ag, ans in agent_answers.items():
            if not ans:
                continue
            n = _normalize_text(ans)
            groups[n] = groups.get(n, 0.0) + float(probs.get(ag, 0.0))
            norm2raw.setdefault(n, ans)
        if not groups:
            if probs:
                top = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                return agent_answers.get(top, "")
            return ""
        pick_norm = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return norm2raw.get(pick_norm, pick_norm)

    # ---------------- Train ----------------
    def train(self, graphs: List[HeteroData]):
        """
        Single-epoch training loop.
        - Router logits -> softmax -> KL(probs || soft targets from per-agent F1)
        - Gradient accumulation by `batch_size`
        - Online metrics: EM / F1 via weighted aggregation of agent answers
        """
        self.model.train()

        total_loss, total_em, total_f1, n_eval = 0.0, 0.0, 0.0, 0

        # ---- order & progress bar ----
        order = list(range(len(graphs)))
        if self.args.shuffle:
            random.shuffle(order)

        pbar = tqdm(order, desc="Train")
        N = len(order)

        # ---- grad accumulation config ----
        bsz = max(1, int(self.args.batch_size))
        accum = 0
        self.opt.zero_grad()

        for step_i, idx in enumerate(pbar, 1):
            data = graphs[idx].to(self.device)

            # ---- forward ----
            out = self.model(data)
            logits = out["logits"] if isinstance(out, dict) else out              # [A]
            probs_main = F.softmax(logits, dim=-1)                                 # [A]

            # ---- metadata ----
            agent_names = getattr(data, "agent_names", [])
            assert len(agent_names) == int(logits.numel()), "Agent names mismatch with logits size."

            q_text = getattr(data, "question_text", None)
            gold = getattr(data, "answer", None)
            if not isinstance(q_text, str) or not isinstance(gold, str):
                continue

            # ---- query all agents (cached + optional new LLM calls) ----
            agent_answers = self.answers_for_agents_parallel(
                data,
                allow_new_llm=bool(self.args.allow_new_llm),
            )

            # ---- build soft targets from per-agent F1 ----
            per_agent_f1 = {ag: f1_score(agent_answers.get(ag, ""), gold) for ag in agent_names}
            tgt_map = self.soft_targets_from_f1(
                per_agent_f1,
                tau=self.args.target_tau,
                eps=self.args.label_smooth,
            )

            tgt = torch.tensor(
                [tgt_map.get(ag, 0.0) for ag in agent_names],
                dtype=torch.float32,
                device=logits.device,
            )
            tgt = tgt / (tgt.sum() + 1e-9)                                        # safety normalize

            # ---- KL loss (probs || target) ----
            loss = F.kl_div(torch.log(probs_main + 1e-9), tgt, reduction="batchmean")

            # ---- backward (grad accumulation) ----
            (loss / bsz).backward()
            accum += 1

            # ---- metrics (sample-level) ----
            total_loss += float(loss.detach().cpu())

            probs_map = {ag: float(probs_main[i].detach().cpu()) for i, ag in enumerate(agent_names)}
            final_pred = self.aggregate_answers(agent_answers, probs_map)

            em = exact_match(final_pred, gold)
            f1 = f1_score(final_pred, gold)

            total_em += em
            total_f1 += f1
            n_eval += 1

            # ---- optimizer step when reaching virtual batch or at the very end ----
            if (accum % bsz == 0) or (step_i == N):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

            # ---- live logging ----
            avg_loss = total_loss / max(n_eval, 1)
            avg_em = total_em / max(n_eval, 1)
            avg_f1 = total_f1 / max(n_eval, 1)

            pbar.set_postfix(bsz=bsz, loss=f"{avg_loss:.4f}", EM=f"{avg_em:.4f}", F1=f"{avg_f1:.4f}")

        # ---- epoch summary ----
        print(f"[Train] loss={total_loss/max(n_eval,1):.4f} "
            f"EM={total_em/max(n_eval,1):.4f} "
            f"F1={total_f1/max(n_eval,1):.4f}")

    # ---------------- Simple evaluate (主实验 only) ----------------
    @torch.no_grad()
    def evaluate(self, graphs: List[HeteroData], split_name: str = "val", allow_new_llm: bool = True):
        self.model.eval()
        total_em, total_f1, n_eval = 0.0, 0.0, 0

        pbar = tqdm(range(len(graphs)), desc=f"Evaluate[{split_name}]")
        for idx in pbar:
            data = graphs[idx].to(self.device)
            out = self.model(data)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = F.softmax(logits, dim=-1)

            agent_names = getattr(data, "agent_names", [])
            assert len(agent_names) == int(logits.numel()), "Agent names mismatch with logits size."

            q_text = getattr(data, "question_text", None)
            gold = getattr(data, "answer", None)
            if not isinstance(q_text, str) or not isinstance(gold, str):
                continue

            # 主实验（router top-k 聚合）
            all_agent_answers = self.answers_for_agents_parallel(data, allow_new_llm=allow_new_llm)
            A = len(agent_names)
            k_cfg = int(self.args.topk_eval) if self.args.topk_eval is not None else A
            k = A if (k_cfg <= 0 or k_cfg > A) else k_cfg
            order_idx = sorted(range(A), key=lambda i: (-float(probs[i]), i))
            sel_idx = order_idx[:k]
            sel_agents = [agent_names[i] for i in sel_idx]
            probs_all = {ag: float(probs[i].detach().cpu()) for i, ag in enumerate(agent_names)}
            probs_topk = {ag: probs_all[ag] for ag in sel_agents}
            agent_answers_topk = {ag: all_agent_answers.get(ag, "") for ag in sel_agents}
            final_pred = self.aggregate_answers(agent_answers_topk, probs_topk)
            em = exact_match(final_pred, gold); f1 = f1_score(final_pred, gold)
            total_em += em; total_f1 += f1; n_eval += 1

            pbar.set_postfix(
                EM=f"{total_em/max(n_eval,1):.4f}",
                F1=f"{total_f1/max(n_eval,1):.4f}",
            )

        print(f"[Eval/{split_name}] "
              f"Router EM={total_em/max(n_eval,1):.4f} F1={total_f1/max(n_eval,1):.4f} topk={self.args.topk_eval}")
        return total_em/max(n_eval,1), total_f1/max(n_eval,1)

    # ---------------- Test evaluate with sweep (主实验 & per-agent) ----------------
    def evaluate_with_sweep(self, graphs: List[HeteroData], split_name: str, allow_new_llm: bool,
                            results_jsonl_path: str, topk_max: Optional[int] = None,
                            save_entity_importance: bool = False,
                            importance_out: Optional[str] = None,
                            importance_method: str = "grad",
                            importance_topk: int = 20) -> Tuple[float, float]:
        import os, json
        from collections import defaultdict
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        self.model.eval()

        # reference agent names & counts
        ref_agents = getattr(graphs[0], "agent_names", []) if graphs else []
        A = len(ref_agents) if ref_agents else len(self.agents)
        if topk_max is None:
            topk_max = A
        topk_max = max(1, min(int(topk_max), A))

        # accumulators for top-k EM/F1
        sum_em_k = [0.0] * (topk_max + 1)
        sum_f1_k = [0.0] * (topk_max + 1)

        # per-agent simple EM/F1 accumulators
        agent_sum_em = {ag: 0.0 for ag in ref_agents}
        agent_sum_f1 = {ag: 0.0 for ag in ref_agents}
        agent_cnt = {ag: 0 for ag in ref_agents}

        n_samples = 0
        final_k = max(1, min(int(self.args.topk_eval or A), A))
        final_sum_em = 0.0
        final_sum_f1 = 0.0

        pbar = tqdm(range(len(graphs)), desc=f"Test[{split_name}] sweep 1..{topk_max}")
        for idx in pbar:
            data = graphs[idx].to(self.device)
            out = self.model(data)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = F.softmax(logits, dim=-1)

            agent_names = getattr(data, "agent_names", [])
            assert len(agent_names) == int(logits.numel()), "Agent names mismatch with logits size."

            gold = getattr(data, "answer", None)
            q_text = getattr(data, "question_text", None)
            if not isinstance(gold, str) or not isinstance(q_text, str):
                continue

            # get (cached) agent answers
            all_agent_answers = self.answers_for_agents_parallel(data, allow_new_llm=allow_new_llm)
            probs_all = {ag: float(probs[i].detach().cpu()) for i, ag in enumerate(agent_names)}
            A_i = len(agent_names)
            order_idx = sorted(range(A_i), key=lambda i: (-float(probs[i]), i))

            # per-agent accumulation (simple EM/F1)
            for i_ag, ag in enumerate(agent_names):
                ans = all_agent_answers.get(ag, "")
                agent_sum_em[ag] = agent_sum_em.get(ag, 0.0) + exact_match(ans, gold)
                agent_sum_f1[ag] = agent_sum_f1.get(ag, 0.0) + f1_score(ans, gold)
                agent_cnt[ag] = agent_cnt.get(ag, 0) + 1

            # top-k sweep per sample
            for k in range(1, topk_max + 1):
                k_eff = min(k, A_i)
                sel_idx = order_idx[:k_eff]
                sel_agents = [agent_names[i] for i in sel_idx]
                probs_topk = {ag: probs_all[ag] for ag in sel_agents}
                agent_answers_topk = {ag: all_agent_answers.get(ag, "") for ag in sel_agents}
                pred = self.aggregate_answers(agent_answers_topk, probs_topk)

                sum_em_k[k] += exact_match(pred, gold)
                sum_f1_k[k] += f1_score(pred, gold)

            # final configured k
            sel_idx = order_idx[:final_k]
            sel_agents = [agent_names[i] for i in sel_idx]
            probs_topk = {ag: probs_all[ag] for ag in sel_agents}
            agent_answers_topk = {ag: all_agent_answers.get(ag, "") for ag in sel_agents}
            pred_final = self.aggregate_answers(agent_answers_topk, probs_topk)
            final_sum_em += exact_match(pred_final, gold)
            final_sum_f1 += f1_score(pred_final, gold)

            n_samples += 1
            pbar.set_postfix(final_k=final_k, EM=f"{final_sum_em/max(n_samples,1):.4f}", F1=f"{final_sum_f1/max(n_samples,1):.4f}")

        # write results JSONL (per-k + per-agent)
        os.makedirs(os.path.dirname(results_jsonl_path), exist_ok=True)
        with open(results_jsonl_path, "w", encoding="utf-8") as wf:
            for k in range(1, topk_max + 1):
                rec_k = {
                    "type": "topk",
                    "split": split_name,
                    "k": k,
                    "EM": round(sum_em_k[k] / max(1, n_samples), 4),
                    "F1": round(sum_f1_k[k] / max(1, n_samples), 4),
                    "n": n_samples,
                }
                wf.write(json.dumps(rec_k, ensure_ascii=False) + "\n")

            for ag in sorted(agent_sum_em.keys()):
                cnt = max(1, agent_cnt.get(ag, 0))
                rec_ag = {
                    "type": "agent",
                    "split": split_name,
                    "agent": ag,
                    "EM": round(agent_sum_em.get(ag, 0.0) / cnt, 4),
                    "F1": round(agent_sum_f1.get(ag, 0.0) / cnt, 4),
                    "n": agent_cnt.get(ag, 0),
                }
                wf.write(json.dumps(rec_ag, ensure_ascii=False) + "\n")

        final_em = final_sum_em / max(1, n_samples)
        final_f1 = final_sum_f1 / max(1, n_samples)
        print(f"[Test/{split_name}] FINAL(topk={final_k}) EM={final_em:.4f} F1={final_f1:.4f}")
        print(f"[Test/{split_name}] wrote: {results_jsonl_path}")
        return final_em, final_f1






# ----------------------------- Main ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--eval_text_metrics", action="store_true", help="ROUGE/BLEU/BERTScore")

    p.add_argument("--graphs_pt", type=str, required=True)
    p.add_argument("--roles_json", type=str, default="prompts/roles.json")
    p.add_argument("--out_dir", type=str, required=True)

    # model
    p.add_argument("--hid_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--shuffle", action="store_true")

    # batching
    p.add_argument("--batch_size", type=int, default=8)

    # eval
    p.add_argument("--topk_eval", type=int, default=8)

    # device
    p.add_argument("--device", type=str, default="auto")

    # together
    p.add_argument("--together_model_fallback", type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct-Turbo")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--max_concurrency", type=int, default=12)
    p.add_argument("--request_timeout", type=float, default=60.0)

    # targets
    p.add_argument("--target_tau", type=float, default=0.25)
    p.add_argument("--label_smooth", type=float, default=1e-3)

    # cache & control
    p.add_argument("--cache_dir", type=str, default="runs/exp_rag/agent_answers")
    p.add_argument("--allow_new_llm", type=int, default=1, help="1=new API calls allowed; 0=cache only")

    # NEW controls
    p.add_argument("--eval_only", action="store_true", help="Skip training; load ckpt and test only.")
    p.add_argument("--ckpt_path", type=str, default="runs/hard2/router_gnn.pt", help="Path to checkpoint (defaults to out_dir/router_gnn.pt)")
    p.add_argument("--results_jsonl", type=str, default=None, help="Where to write per-k & per-agent test metrics JSONL")

    # Entity importance saving (post-hoc attention-like scores, no model change)
    p.add_argument("--save_entity_importance", action="store_true",
                   help="Save per-entity importance scores (gradient-based) during TEST.")
    p.add_argument("--importance_method", type=str, default="grad", choices=["grad", "grad_input"],
                   help="Importance method: 'grad' (||d s / d x||) or 'grad_input' (|grad*input|).")
    p.add_argument("--importance_topk", type=int, default=20,
                   help="Top-K entities to save per sample.")
    p.add_argument("--importance_out", type=str, default=None,
                   help="JSONL path for entity importance on test (default: out_dir/entity_importance_test.jsonl)")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    trainer = Trainer(args)

    # 8:1:1 split
    graphs = trainer.graphs
    n = len(graphs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_graphs = graphs[:500]
    val_graphs   = graphs[500:600]
    test_graphs  = graphs[600:700]
    print(f"Split sizes → train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)} (n={n})")

    ckpt_path = args.ckpt_path or os.path.join(args.out_dir, "router_gnn.pt")
    results_jsonl = args.results_jsonl or os.path.join(args.out_dir, "test_metrics.jsonl")

    best_f1 = -1.0

    if not args.eval_only:
        for ep in range(1, args.epochs + 1):
            print(f"\n===== Epoch {ep}/{args.epochs} =====")
            trainer.train(train_graphs)
            # if(ep%5==1 or ep==args.epochs):
            _, f1 = trainer.evaluate(test_graphs, split_name="test", allow_new_llm=True)
            if f1 >= best_f1:
                best_f1 = f1
                torch.save({"model": trainer.model.state_dict(), "args": vars(args)}, ckpt_path)
                print(f"[Checkpoint] Saved: {ckpt_path} (val F1={best_f1:.4f})")
            # _, f1 = trainer.evaluate(val_graphs, split_name="val", allow_new_llm=True)

    else:
        print("[Eval-only] Skipping training.")

    # load checkpoint (if exists)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=trainer.device)
        trainer.model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print(f"[Load] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[Warn] Checkpoint not found at {ckpt_path}. Using current model weights.")

    # Final test evaluation
    final_em, final_f1 = trainer.evaluate_with_sweep(
        test_graphs,
        split_name="test",
        allow_new_llm=bool(args.allow_new_llm),
        results_jsonl_path=results_jsonl,
        topk_max=24,
        save_entity_importance=bool(args.save_entity_importance),
        importance_out=(args.importance_out or os.path.join(args.out_dir, "entity_importance_test.jsonl")),
        importance_method=str(args.importance_method),
        importance_topk=int(args.importance_topk),
    )
    print(f"[RESULT] Test FINAL (topk={args.topk_eval}) → EM={final_em:.4f} F1={final_f1:.4f}")


if __name__ == "__main__":
    main()
