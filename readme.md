# NGRouter

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Environment](#environment)
- [Build Graph](#build-graph)
- [Train Router](#train-router)

---

## Overview

### Nodes

- `question`: one node per sample
- `entity`: extracted by NER from context text
- `agent`: fixed 24 roles 

### Edges

- **Static**
  - `entity - entity` (rel)
  - `question - entity` (q_ref)
  - `agent - entity` (agent_manage)
- **Trainable**
  - `question → agent` (q2agent) and reverse (q2agent_rev)

---

### Objective

- Learn the routing distribution from question → agent to improve answer accuracy.

---

## Dependencies

### Step 1: Create Virtual Environment

```bash
conda create -n GraphRAG python=3.11 -y
conda activate GraphRAG
```

### Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 3: Download spaCy Model

```bash
python -m spacy download en_core_web_trf
```

---

## Environment

Set up environment variables for LLM usage:

```bash
export OPENAI_API_KEY="sk-..."   #OpenAI API Key
```

## Build Graph

### Process Graph
```bash
python data/scripts/process_ngqa.py --csv data/NGQA_benchmark --output data/NGQA_benchmark --limit 1000
```

### Build Graph Example

```bash
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
```

```bash
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
```
### Output Files

- out_json/ — one JSON graph per sample
- out_pyg.pt — list of PyG HeteroData graphs for all samples in the split

---

## Train Router

```bash
export TOGETHER_API_KEY="" # Your Together API KEY
```

```bash
    python training/train_graphrag.py \
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
    --save_entity_importance
```



