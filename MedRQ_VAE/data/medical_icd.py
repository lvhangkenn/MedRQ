import os
import json
import torch
from typing import Optional


class MedicalICD:

    def __init__(self, root: str, force_process: bool = False, *args, **kwargs) -> None:
        self.root = root
        self.processed_dir = os.path.join(self.root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_paths = [os.path.join(self.processed_dir, "medical.pt")]
        self.data = {}

        if os.path.exists(self.processed_paths[0]) and not force_process:
            self.data = torch.load(self.processed_paths[0], map_location="cpu", weights_only=False)
            return

        self.process()

    @property
    def processed_paths(self): 
        return self._processed_paths

    @processed_paths.setter
    def processed_paths(self, value):  
        self._processed_paths = value

    def _load_optional_texts(self, n_items: int):
        items_path = os.path.join(self.root, "icd_items.jsonl")
        if not os.path.exists(items_path):
            return None
        texts = []
        try:
            with open(items_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    texts.append(rec.get("text", ""))
        except Exception:
            return None
        if len(texts) != n_items:
            return None
        return texts

    def process(self, max_seq_len: Optional[int] = None) -> None:
    
        x_path = os.path.join(self.root, "embeddings", "icd_text_emb.pt")
        idx_path = os.path.join(self.root, "tags_indices.pt")
        if not (os.path.exists(x_path) and os.path.exists(idx_path)):
            raise FileNotFoundError("Missing required medical artifacts under {}".format(self.root))

        x = torch.load(x_path, map_location="cpu", weights_only=False).float()
        tags_indices = torch.load(idx_path, map_location="cpu", weights_only=False).long()
        n_items, emb_dim = x.shape[0], x.shape[1]

    
        layer_files = [
            ("l2", os.path.join(self.root, "embeddings", "tags_emb_l2.pt")),
            ("l3", os.path.join(self.root, "embeddings", "tags_emb_l3.pt")),
            ("l4", os.path.join(self.root, "embeddings", "tags_emb_l4.pt")),
            ("l5", os.path.join(self.root, "embeddings", "tags_emb_l5.pt")),
        ]
        layer_embs = []
        for _, f in layer_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f)
            e = torch.load(f, map_location="cpu", weights_only=False).float()
            layer_embs.append(e)

        layers = tags_indices.shape[1]
        tags_emb = torch.zeros((n_items, layers, emb_dim), dtype=torch.float32)
        for k in range(layers):
            idx_k = tags_indices[:, k]
            valid = idx_k >= 0
            if valid.any():
                selected = layer_embs[k].index_select(0, idx_k[valid].to(torch.long))  # [Nv, emb_dim]
                tags_emb[valid, k, :] = selected
        gen = torch.Generator()
        gen.manual_seed(42)
        is_train = (torch.rand(n_items, generator=gen) > 0.05)

        texts = self._load_optional_texts(n_items)

        self.data = {
            "item": {
                "x": x,  # [N, 768]
                "tags_indices": tags_indices,  # [N, 4]
                "tags_emb": tags_emb,  # [N, 4, 768]
                "is_train": is_train,  # [N]
            }
        }
        if texts is not None:
            self.data["item"]["text"] = texts

        torch.save(self.data, self.processed_paths[0])


