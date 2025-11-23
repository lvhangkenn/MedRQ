import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PAD_TOKEN = "__PAD__"


def ensure_tmpdir() -> None:
    """Force dill/tmp usage into a writable directory."""
    if "TMPDIR" not in os.environ or not os.environ["TMPDIR"]:
        os.environ["TMPDIR"] = "."


def ensure_dill_import() -> None:
    """Make dill able to load python2-era pickles."""
    ensure_tmpdir()
    import builtins  # lazy import to avoid unnecessary dependency setup

    sys.modules.setdefault("__builtin__", builtins)


def load_voc_codes(voc_path: str, key: str) -> List[str]:
    """Load ordered codes from MedAlign voc pickle via dill."""
    ensure_dill_import()
    import dill  # type: ignore

    with open(voc_path, "rb") as fp:
        voc_bundle = dill.load(fp)
    voc = voc_bundle[key]
    return [str(voc.idx2word[i]) for i in range(len(voc.idx2word))]


def default_norm(code: Optional[str]) -> str:
    if code is None:
        return ""
    return "".join(ch for ch in str(code).upper() if ch.isalnum())


def norm_with_keep_letters(code: Optional[str]) -> str:
    if code is None:
        return ""
    return "".join(ch for ch in str(code).upper().strip() if not ch.isspace())


@dataclass
class OntologyRecord:
    code_norm: str
    parent_norm: str
    name: str
    raw_code: str


def load_ontology_table(
    csv_path: str,
    code_col: str = "code",
    parent_col: str = "parent_code",
    name_col: str = "name",
    norm_fn: Callable[[Optional[str]], str] = default_norm,
) -> Dict[str, OntologyRecord]:
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str)
    table: Dict[str, OntologyRecord] = {}
    for rec in df.to_dict(orient="records"):
        raw_code = rec.get(code_col)
        if raw_code is None:
            continue
        raw_code_str = str(raw_code).strip()
        if not raw_code_str or raw_code_str.lower() == "nan":
            continue
        code_norm = norm_fn(raw_code)
        if not code_norm:
            continue
        parent_val = rec.get(parent_col)
        parent_norm = ""
        if parent_val is not None:
            parent_str = str(parent_val).strip()
            if parent_str and parent_str.lower() != "nan":
                parent_norm = norm_fn(parent_val)
        name_val = rec.get(name_col)
        name = ""
        if isinstance(name_val, str):
            name_clean = name_val.strip()
            if name_clean and name_clean.lower() != "nan":
                name = name_clean
        table[code_norm] = OntologyRecord(
            code_norm=code_norm,
            parent_norm=parent_norm,
            name=name,
            raw_code=str(raw_code) if raw_code is not None else code_norm,
        )
    return table


def lookup_name(table: Dict[str, OntologyRecord], code_norm: str) -> str:
    rec = table.get(code_norm)
    if rec is None:
        return code_norm
    return rec.name if rec.name else rec.raw_code


def build_path_codes(
    table: Dict[str, OntologyRecord],
    code_norm: str,
    max_hops: int = 20,
) -> List[str]:
    path: List[str] = []
    current = code_norm
    seen = set()
    depth = 0
    while current and current not in seen and depth < max_hops:
        seen.add(current)
        path.append(current)
        rec = table.get(current)
        if rec is None or not rec.parent_norm:
            break
        current = rec.parent_norm
        depth += 1
    return list(reversed(path))


def extract_tag_sequence(
    full_names: Sequence[str],
    level_count: int,
    drop_root: bool,
    drop_leaf: bool,
    pad_direction: str,
    fallback_leaf: Optional[str] = None,
) -> List[str]:
    seq = list(full_names)
    if drop_root and seq:
        seq = seq[1:]
    if drop_leaf and seq:
        seq = seq[:-1]
    seq = [s for s in seq if s]
    if not seq and fallback_leaf:
        seq = [fallback_leaf]
    if len(seq) > level_count:
        if pad_direction == "front":
            seq = seq[-level_count:]
        else:
            seq = seq[:level_count]
    if len(seq) < level_count:
        pad_needed = level_count - len(seq)
        if pad_direction == "front":
            seq = [PAD_TOKEN] * pad_needed + seq
        else:
            seq = seq + [PAD_TOKEN] * pad_needed
    return seq


def write_vocab_order(codes: Sequence[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for code in codes:
            f.write(str(code) + "\n")


def write_items_jsonl(
    codes: Sequence[str],
    table: Dict[str, OntologyRecord],
    norm_fn: Callable[[Optional[str]], str],
    seqs: Sequence[Sequence[str]],
    path: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for code, tags in zip(codes, seqs):
            cn = norm_fn(code)
            text = lookup_name(table, cn)
            rec = {"code": str(code), "text": text, "tags": list(tags)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_tag_vocabs(
    tag_sequences: Sequence[Sequence[str]],
    level_labels: Sequence[str],
    out_dir: str,
) -> List[List[str]]:
    vocabs: List[List[str]] = []
    for idx, label in enumerate(level_labels):
        values = {tag_sequences[row][idx] for row in range(len(tag_sequences))}
        # 不包含 PAD_TOKEN，直接排序所有非空值
        tokens = sorted(v for v in values if v and v != PAD_TOKEN)
        vocab_path = os.path.join(out_dir, f"tag_vocab_{label}.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in tokens:
                f.write(token + "\n")
        vocabs.append(tokens)
    return vocabs


def build_tag_indices_tensor(
    tag_sequences: Sequence[Sequence[str]],
    vocabs: Sequence[Sequence[str]],
) -> "torch.Tensor":
    import torch

    vocab_maps = [{tok: i for i, tok in enumerate(vocab)} for vocab in vocabs]
    rows = len(tag_sequences)
    cols = len(vocabs)
    idx = torch.full((rows, cols), fill_value=-1, dtype=torch.long)
    for i, tags in enumerate(tag_sequences):
        for j, tag in enumerate(tags):
            tag_id = vocab_maps[j].get(tag)
            if tag_id is not None:
                idx[i, j] = tag_id
    return idx


def save_tensor(tensor: "torch.Tensor", path: str) -> None:
    import torch

    torch.save(tensor, path)


def encode_texts_with_biobert(
    texts: Sequence[str],
    device: str = "cuda",
    batch_size: int = 64,
    model_name: str = "dmis-lab/biobert-v1.1",
) -> "torch.Tensor":
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            last_hidden = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(pooled.cpu())
    if outputs:
        return torch.cat(outputs, dim=0)
    import torch as torch_module

    return torch_module.empty(0, mdl.config.hidden_size)


def save_embeddings(
    texts: Sequence[str],
    out_path: str,
    device: str = "cuda",
    batch_size: int = 64,
    model_name: str = "dmis-lab/biobert-v1.1",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    embeddings = encode_texts_with_biobert(
        texts, device=device, batch_size=batch_size, model_name=model_name
    )
    save_tensor(embeddings, out_path)


def load_vocab_tokens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--voc-path", type=str, required=True, help="voc_final.pkl path")
    parser.add_argument(
        "--ontology-csv", type=str, required=True, help="Ontology CSV with code-parent"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory (will be created if missing)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="If set, generate BioBERT embeddings for items and tags",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embeddings (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embeddings (default: 64)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dmis-lab/biobert-v1.1",
        help="HuggingFace model identifier for BioBERT",
    )
    return parser
