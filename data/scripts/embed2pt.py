#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embed JSON graphs with BERT (768-dim) into PyG HeteroData and save as a .pt list.

- Keeps read order (no reordering)
- Uses EXACT edges from JSON (entity↔entity, question↔entity, agent↔entity)
- BERT-only embeddings (768 dims)
- Entity embeddings include node type: "<name> [SEP] type: <node type>"

Usage:
    python embed_json_to_pt_bert_exact_edges_with_types.py \
            --json_dir path/to/json_graphs \
            --out_pt   path/to/output/graphs.pt \
            --model_name bert-base-uncased \
            --pooling mean \
            --max_length 256 \
            --batch_size 64 \
            --device auto \
            --file_suffix .graph.json \
            --verify_n 5
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ----------------------------
# Args
# ----------------------------
def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--json_dir", type=str, required=True, help="Folder containing *.graph.json")
        p.add_argument("--out_pt", type=str, required=True, help="Output .pt path")
        p.add_argument("--model_name", type=str, default="bert-base-uncased")
        p.add_argument("--pooling", type=str, choices=["mean", "cls", "first_last_avg"], default="mean")
        p.add_argument("--max_length", type=int, default=256)
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--device", type=str, default="auto", help="'auto'|'cpu'|'cuda'|'mps'")
        p.add_argument("--file_suffix", type=str, default=".json", help="File suffix to load")
        p.add_argument("--verify_n", type=int, default=5, help="Print N samples for sanity check")
        return p.parse_args()

def extract_question_id_from_path(fp: str) -> Optional[int]:
        """
        Extract the numeric index from a filename and convert it to an integer.
        Example: 'graph_000097.json' -> 97
        Returns None if no digits are found.
        """
        base = os.path.basename(fp)
        m = re.search(r"(\d+)", base)
        if not m:
                return None
        # Strip leading zeros, fallback to 0
        return int(m.group(1).lstrip("0") or "0")

# ----------------------------
# BERT Embedder (768-dim)
# ----------------------------
class HFBertEmbedder:
        def __init__(self, model_name: str, pooling: str, max_length: int, batch_size: int, device: Optional[str]):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(
                        model_name,
                        output_hidden_states=(pooling == "first_last_avg")
                )
                self.model.eval()

                if device is None or device == "auto":
                        if torch.cuda.is_available():
                                device = "cuda"
                        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                                device = "mps"
                        else:
                                device = "cpu"
                self.device = device
                self.model.to(self.device)

                self.pooling = pooling
                self.max_length = max_length
                self.batch_size = batch_size

        @torch.inference_mode()
        def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
                if isinstance(texts, str):
                        texts = [texts]
                all_emb = []
                for i in range(0, len(texts), self.batch_size):
                        batch = texts[i:i + self.batch_size]
                        tok = self.tokenizer(
                                batch, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt"
                        )
                        tok = {k: v.to(self.device) for k, v in tok.items()}
                        out = self.model(**tok)

                        if self.pooling == "cls":
                                if getattr(out, "pooler_output", None) is not None and out.pooler_output is not None:
                                        emb = out.pooler_output
                                else:
                                        emb = out.last_hidden_state[:, 0, :]
                        elif self.pooling == "first_last_avg":
                                hs = out.hidden_states
                                mask = tok["attention_mask"].unsqueeze(-1).float()
                                first = (hs[1] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                                last = (hs[-1] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                                emb = 0.5 * (first + last)
                        else:  # mean pooling
                                mask = tok["attention_mask"].unsqueeze(-1).float()
                                seq = out.last_hidden_state
                                emb = (seq * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

                        if normalize:
                                emb = F.normalize(emb, p=2, dim=1)

                        all_emb.append(emb.detach().cpu())
                return torch.cat(all_emb, dim=0)  # [N, 768]


# ----------------------------
# Utils
# ----------------------------
def collect_json_files(json_dir: str, suffix: str) -> List[str]:
        files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(suffix)]
        return sorted(files)


def norm_rel(rel: str) -> str:
        rel = (rel or "").strip().lower()
        rel = re.sub(r"\s+", "_", rel)
        rel = re.sub(r"[^a-z0-9_\:\-\.]+", "", rel)
        return f"rel:{rel if rel else 'rel'}"


# ----------------------------
# Build HeteroData (use EXACT JSON edges; entity includes node type)
# ----------------------------
def build_heterodata_from_json(
        g: Dict[str, Any],
        embedder: HFBertEmbedder,
) -> HeteroData:
        # ----- meta / question text / answer -----
        meta = g.get("meta", {}) or {}
        q_text = meta.get("question", "")
        answer = meta.get("answer", None)

        # prefer meta.question; fallback to first item in "question" section
        q_nodes = g.get("question", []) or []
        if not q_text and isinstance(q_nodes, list) and len(q_nodes) > 0:
                q_text = q_nodes[0].get("question", "") or ""
        # question node name for edge name resolution
        q_node_name = None
        if isinstance(q_nodes, list) and len(q_nodes) > 0:
                q_node_name = q_nodes[0].get("node name", None) or "question"
        else:
                q_node_name = "question"  # safe default

        # ----- nodes -----
        entities: List[Dict[str, Any]] = g.get("nodes", []) or []
        agents:   List[Dict[str, Any]] = g.get("agents", []) or []
        edges:    List[Dict[str, Any]] = g.get("edges", []) or []

        entity_names = [e.get("node name", "") for e in entities]
        entity_types = [e.get("node type", "") for e in entities]  # <- keep original node type strings
        agent_names  = [a.get("node name", "") for a in agents]

        # ----- embeddings (BERT 768) -----
        # Entity: include node type in text
        entity_texts = [
                f"{e.get('node name','')} [SEP] type: {e.get('node type','')}"
                for e in entities
        ]
        agent_texts = [
                " | ".join([
                        a.get("node name", ""),
                        a.get("node type", ""),
                        str(a.get("backbone_id", "")),
                        (a.get("prompt", "") or "")[:512],
                ]) for a in agents
        ]

        texts = []
        span_q = (0, 1); texts.append(q_text)
        span_e = (len(texts), len(entity_texts)); texts.extend(entity_texts)
        span_a = (len(texts), len(agent_texts));  texts.extend(agent_texts)

        embs = embedder.encode(texts, normalize=True)  # [N, 768]
        q_x = embs[span_q[0]: span_q[0] + span_q[1]]              # [1, 768]
        e_x = embs[span_e[0]: span_e[0] + span_e[1]] if span_e[1] > 0 else torch.zeros((0, 768))
        a_x = embs[span_a[0]: span_a[0] + span_a[1]] if span_a[1] > 0 else torch.zeros((0, 768))

        data = HeteroData()
        data["question"].x = q_x
        data["entity"].x   = e_x
        data["agent"].x    = a_x

        # (optional) keep original strings for reference/analysis
        data.entity_names = entity_names
        data.entity_types = entity_types  # <- store the node type strings
        data.agent_names  = agent_names

        # ----- name -> (type, idx) resolver -----
        name2idx_type: Dict[str, Tuple[str, int]] = {}
        # question (single node, idx 0)
        if q_node_name:
                name2idx_type[q_node_name] = ("question", 0)
        # entities
        for i, nm in enumerate(entity_names):
                if nm:
                        name2idx_type[nm] = ("entity", i)
        # agents
        for i, nm in enumerate(agent_names):
                if nm:
                        name2idx_type[nm] = ("agent", i)

        # ----- materialize EXACT edges -----
        buckets: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
        for rec in edges:
                s_name = rec.get("source", "")
                t_name = rec.get("target", "")
                rel    = norm_rel(rec.get("relation", "rel"))

                s_info = name2idx_type.get(s_name)
                t_info = name2idx_type.get(t_name)
                if s_info is None or t_info is None:
                        # silently skip dangling names; switch to a warning if you prefer
                        continue

                s_type, s_idx = s_info
                t_type, t_idx = t_info
                key = (s_type, rel, t_type)
                buckets.setdefault(key, []).append((s_idx, t_idx))

        for key, pairs in buckets.items():
                src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
                dst_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
                data[key].edge_index = torch.stack([src_idx, dst_idx], dim=0)

        # ----- attach handy fields (binding preserved by order) -----
        data.answer = answer
        data.question_text = q_text
        data.question_node_name = q_node_name
        data.meta = meta
        data.context = g.get("context", None)

        return data


# ----------------------------
# Main
# ----------------------------
def main():
        args = parse_args()
        os.makedirs(os.path.dirname(os.path.abspath(args.out_pt)), exist_ok=True)

        files = collect_json_files(args.json_dir, args.file_suffix)
        if not files:
                raise FileNotFoundError(f"No *{args.file_suffix} found under {args.json_dir}")

        embedder = HFBertEmbedder(
                model_name=args.model_name,
                pooling=args.pooling,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=(None if args.device == "auto" else args.device),
        )

        graphs: List[HeteroData] = []
        pbar = tqdm(total=len(files), desc="Embed JSON graphs (BERT 768, exact edges + entity types)", unit="file")
        for fp in files:
                try:
                        with open(fp, "r", encoding="utf-8") as f:
                                g = json.load(f)
                        data = build_heterodata_from_json(g, embedder)

                        # ← Added: extract question_id from filename and attach it to data
                        qid = extract_question_id_from_path(fp)
                        data.qid = qid

                        graphs.append(data)

                except Exception as e:
                        print(f"[WARN] Failed on {fp}: {e}")
                pbar.update(1)
        pbar.close()

        torch.save(graphs, args.out_pt)
        print(f"[OK] Saved {len(graphs)} graphs -> {args.out_pt}")

        # quick verify
        ver_n = max(0, int(args.verify_n))
        for i in range(min(ver_n, len(graphs))):
                d = graphs[i]
                first_ent_type = (d.entity_types[0] if isinstance(getattr(d, "entity_types", None), list) and len(d.entity_types)>0 else None)
                print("\n[VERIFY]", {
                        "i": i,
                        "question": (getattr(d, "question_text", "")[:120] + "...") if isinstance(getattr(d, "question_text", ""), str) and len(getattr(d, "question_text", "")) > 120 else getattr(d, "question_text", ""),
                        "answer": getattr(d, "answer", None),
                        "num_entities": d["entity"].num_nodes if "entity" in d.node_types else 0,
                        "num_agents": d["agent"].num_nodes if "agent" in d.node_types else 0,
                        "edge_types": d.edge_types,
                        "first_entity_type": first_ent_type,
                        "context_len": (len(d.context) if isinstance(getattr(d, "context", None), list) else None),  # <<< NEW
                })


if __name__ == "__main__":
        main()
