import argparse
import json
import os
from typing import List

from medrq_prep_utils import (
    build_path_codes,
    build_tag_indices_tensor,
    build_tag_vocabs,
    norm_with_keep_letters,
    extract_tag_sequence,
    load_ontology_table,
    load_voc_codes,
    lookup_name,
    make_cli_parser,
    save_tensor,
    write_items_jsonl,
    write_vocab_order,
)


"""
python prepare_mimic4_drug_for_hidvae.py \\
  --voc-path ../mimic-iv_data/voc_final.pkl \\
  --ontology-csv ../raw/drug_atc_ontology.csv \\
  --out-dir ../semantic_mimic4/drug \\
  --embed \\
  --device cuda
"""


LEVEL_LABELS = ["L1", "L2", "L3", "L4"]


def parse_args() -> argparse.Namespace:
    parser = make_cli_parser()
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum parent traversal depth when building hierarchies",
    )
    return parser.parse_args()


def build_tag_sequences(
    codes: List[str],
    table,
    norm_fn,
    max_depth: int,
) -> List[List[str]]:
    """
    采用旧版逻辑：直接取 path[0..3]（ATC从根开始，取L1-L4）。
    若路径长度不足，重复末节点填充至4层。
    """
    sequences: List[List[str]] = []
    for code in codes:
        cn = norm_fn(code)
        path_codes = build_path_codes(table, cn, max_hops=max_depth)
        # path_codes: [root(L1), L2, L3, L4/code_norm]

        labels = []
        for i in range(4):  # 取 path[0..3] 作为 L1-L4
            if i < len(path_codes):
                node = path_codes[i]
            else:
                # 不足时重复最后一个节点
                node = path_codes[-1] if path_codes else cn
            labels.append(lookup_name(table, node))

        sequences.append(labels)
    return sequences


def maybe_dump_tag_counts(vocabs, out_dir: str) -> None:
    counts = [len(v) for v in vocabs]
    path = os.path.join(out_dir, "tag_class_counts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(counts, f)


def main() -> None:
    args = parse_args()

    # 1. 从 MIMIC-IV `voc_final.pkl` 中读取 drug 代码（与 med_voc.idx2word 对齐）
    codes = load_voc_codes(args.voc_path, "med_voc")

    # 2. 加载 ATC 本体表
    #    drug_atc_ontology.csv: code, parent_code, name
    table = load_ontology_table(
        args.ontology_csv,
        code_col="code",
        parent_col="parent_code",
        name_col="name",
        norm_fn=norm_with_keep_letters,
    )

    # 3. 构造标签序列（L1-L4）
    tag_sequences = build_tag_sequences(codes, table, norm_with_keep_letters, args.max_depth)

    # 4. 写出对齐的 code 顺序与 item 信息
    os.makedirs(args.out_dir, exist_ok=True)
    vocab_order_path = os.path.join(args.out_dir, "drug_vocab_order.txt")
    write_vocab_order(codes, vocab_order_path)

    items_path = os.path.join(args.out_dir, "drug_items.jsonl")
    write_items_jsonl(codes, table, norm_with_keep_letters, tag_sequences, items_path)

    # 5. 构建标签词表与索引矩阵
    vocabs = build_tag_vocabs(tag_sequences, LEVEL_LABELS, args.out_dir)
    maybe_dump_tag_counts(vocabs, args.out_dir)

    tag_idx = build_tag_indices_tensor(tag_sequences, vocabs)
    # 兼容 medical_icd.py 的命名（无前缀）
    save_tensor(tag_idx, os.path.join(args.out_dir, "tags_indices.pt"))

    # 6. 可选：生成文本与标签的 BioBERT embedding
    if args.embed:
        from hidvae_prep_utils import load_vocab_tokens, save_embeddings

        items = []
        with open(items_path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        texts = [rec["text"] if rec["text"] else rec["code"] for rec in items]
        save_embeddings(
            texts,
            os.path.join(args.out_dir, "embeddings", "drug_text_emb.pt"),
            device=args.device,
            batch_size=args.batch_size,
            model_name=args.model_name,
        )

        for label in LEVEL_LABELS:
            vocab_tokens = load_vocab_tokens(
                os.path.join(args.out_dir, f"tag_vocab_{label}.txt")
            )
            save_embeddings(
                vocab_tokens,
                os.path.join(args.out_dir, "embeddings", f"tags_emb_{label}.pt"),
                device=args.device,
                batch_size=args.batch_size,
                model_name=args.model_name,
            )


if __name__ == "__main__":
    main()


