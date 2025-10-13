# -*- coding: utf-8 -*-
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData


class RouterGNN(nn.Module):
    """
    面向你的 JSON 图的精简版路由模型（无 question-type）：
      - 节点类型：question(单个)、entity(若干)、agent(若干)
      - 边类型：
          * entity --rel:<name>--> entity   （来自 JSON edges 中的 relation）
          * question --qref--> entity       （按问题中提及到的实体名做匹配，或外部传入）
          * agent --manage--> entity        （简单连接策略；见构图代码）
        上述边全部自动加反向边（便于 HGT 双向传播）。
      - 表示学习：各类型节点线性投影 -> 若干层 HGTConv
      - 路由打分：对每个 agent，score = MLP(tanh( (Wq*q) ⊙ (Wa*a) ))
      - 训练：对 logits 做 softmax，与你的 per-agent 目标分布（如基于 F1/EM 归一化）做 KL
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 256,
        num_layers: int = 2,
        metadata: tuple | None = None,
        dropout: float = 0.2,
        use_instance_norm: bool = False,
        logit_noise_std: float = 0.0,
    ):
        super().__init__()
        self.use_instance_norm = bool(use_instance_norm)
        self.logit_noise_std = float(logit_noise_std)

        # --- 线性输入投影 ---
        self.lin_q = Linear(in_dim, hid_dim)
        self.lin_e = Linear(in_dim, hid_dim)
        self.lin_a = Linear(in_dim, hid_dim)

        # --- HGTConv 堆叠 ---
        if metadata is None:
            # metadata 需要由构图阶段给出：([node_types], [edge_types])
            # edge_types 形如 ("entity","rel:belongs_to","entity") / ("question","qref","entity") 等
            raise ValueError("RouterGNN requires 'metadata' from your HeteroData.")
        self.convs = nn.ModuleList([
            HGTConv(in_channels=hid_dim, out_channels=hid_dim, metadata=metadata, heads=2)
            for _ in range(num_layers)
        ])

        # --- 路由打分头（question × agent） ---
        self.dropout = nn.Dropout(p=dropout)
        self.head_q = nn.Linear(hid_dim, hid_dim)
        self.head_a = nn.Linear(hid_dim, hid_dim)
        self.scorer = nn.Linear(hid_dim, 1)

    # ---- 内部：逐层跑 HGT（含 MPS fallback）----
    def _apply_dropout_dict(self, x_dict):
        return {k: self.dropout(v) for k, v in x_dict.items()}

    def _run_convs(self, x_dict, edge_index_dict, device_str: str):
        try:
            out = x_dict
            for conv in self.convs:
                out = conv(out, edge_index_dict)
                out = self._apply_dropout_dict(out)
            return out
        except NotImplementedError:
            # 某些环境下 HGTConv 在 MPS 不支持，自动转 CPU 再转回
            if device_str == "mps":
                x_cpu = {k: v.cpu() for k, v in x_dict.items()}
                ei_cpu = {k: v.cpu() for k, v in edge_index_dict.items()}
                out = x_cpu
                for conv in self.convs:
                    out = conv(out, ei_cpu)
                    out = self._apply_dropout_dict(out)
                return {k: v.to(device_str) for k, v in out.items()}
            else:
                raise

    def forward(self, data: HeteroData):
        dev = data["question"].x.device
        device_str = dev.type

        # 线性投影
        x_dict = {
            "question": self.lin_q(data["question"].x),  # [1, H]
            "entity":   self.lin_e(data["entity"].x),    # [E, H]
            "agent":    self.lin_a(data["agent"].x),     # [A, H]
        }
        # HGTConv
        x_dict = self._run_convs(x_dict, data.edge_index_dict, device_str=device_str)

        # 路由打分：q × a
        q = x_dict["question"]           # [1, H]
        a = x_dict["agent"]              # [A, H]
        qh = self.head_q(q).expand_as(a) # [A, H]
        ah = self.head_a(a)              # [A, H]
        h = torch.tanh(qh * ah)
        h = self.dropout(h)
        logits = self.scorer(h).squeeze(-1)  # [A]

        if self.use_instance_norm:
            mu = logits.mean()
            sigma = logits.std(unbiased=False) + 1e-6
            logits = (logits - mu) / sigma

        if self.training and self.logit_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.logit_noise_std

        return {"logits": logits}  # 训练时对 logits 做 softmax + KL
