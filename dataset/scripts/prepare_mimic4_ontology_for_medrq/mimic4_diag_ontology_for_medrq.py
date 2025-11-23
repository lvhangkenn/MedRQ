import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from medrq_prep_utils import (
    PAD_TOKEN,
    build_tag_indices_tensor,
    build_tag_vocabs,
    load_voc_codes,
    make_cli_parser,
    save_tensor,
    write_vocab_order,
)


LEVEL_LABELS = ["l2", "l3", "l4", "l5"]


def norm_code(code: Optional[str]) -> str:
    """与 combined.csv 中的 code_norm 对齐：去空格/转大写/去点号。"""
    if code is None:
        return ""
    return str(code).strip().upper().replace(".", "")


class AmbiguityLogger:
    """记录同一 code_norm 多体系候选时的选择信息，便于事后审查。"""

    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []

    @staticmethod
    def _fmt_alts(records: List[dict]) -> str:
        out = []
        for r in records:
            sys = (r.get("system") or "").strip()
            raw = (r.get("code") or "").strip()
            cn = (r.get("code_norm") or "").strip()
            nm = (r.get("name") or "").strip()
            out.append(f"{sys}:{raw}|{cn}|{nm}")
        return "; ".join(out)

    def log(
        self,
        code_raw: str,
        code_norm: str,
        chosen_system: str,
        alternatives: List[dict],
        reason: str,
    ) -> None:
        self.rows.append(
            {
                "code_raw": (code_raw or "").strip(),
                "code_norm": (code_norm or "").strip().upper(),
                "chosen_system": chosen_system or "",
                "alternatives": self._fmt_alts(alternatives),
                "reason": reason,
            }
        )

    def dump_csv(self, out_path: str) -> None:
        if not self.rows:
            pd.DataFrame(
                columns=["code_raw", "code_norm", "chosen_system", "alternatives", "reason"]
            ).to_csv(out_path, index=False)
            return
        pd.DataFrame(self.rows).to_csv(out_path, index=False, encoding="utf-8")


def load_combined_table(csv_path: str):
    """
    从 icd9_icd10_combined.csv 中加载两套体系的记录，并构建多种索引：
      - rows_by_norm[code_norm] -> 所有体系的记录列表
      - rec_by_code[raw_code]   -> 原始 code 对应的记录列表
    """
    df = pd.read_csv(
        csv_path, dtype=str, keep_default_na=False, on_bad_lines="skip", engine="python"
    )
    required = ["system", "code", "code_norm", "parent_code", "parent_code_norm", "name"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in combined CSV: {miss}")

    rows_by_norm: Dict[str, List[dict]] = defaultdict(list)
    rec_by_code: Dict[str, List[dict]] = defaultdict(list)

    for rec in df.to_dict(orient="records"):
        rec["system"] = (rec.get("system") or "").strip()
        rec["code"] = (rec.get("code") or "").strip()
        rec["name"] = (rec.get("name") or "").strip()
        rec["code_norm"] = norm_code(rec.get("code_norm"))
        rec["parent_code_norm"] = norm_code(rec.get("parent_code_norm"))

        cn = rec["code_norm"]
        sys = rec["system"]
        cr = rec["code"].upper()
        if not cn or not sys:
            continue
        rows_by_norm[cn].append(rec)
        rec_by_code[cr].append(rec)

    return rows_by_norm, rec_by_code


# =============================
# EV-aware 体系推断与 V 码父范围启发
# =============================

_V_PURE_RANGE_RE = re.compile(r"^V\d{2}\s*[-–]\s*V\d{2}$", re.IGNORECASE)
_V_DECIMAL_IN_RANGE_RE = re.compile(
    r"^V\d{2}(?:\.\d+)?\s*[-–]\s*V\d{2}(?:\.\d+)?$", re.IGNORECASE
)


def _classify_v_parent_span(raw_parent: str) -> Optional[str]:
    """
    识别 V 码父范围形态：
      - 'Vdd-Vdd'（无小数）       → 'icd10_span'
      - 'Vdd-Vdd.xx' / 'Vdd.xx-Vdd.xx'（任一侧带小数） → 'icd9_span'
    返回 'icd10_span' / 'icd9_span' / None
    """
    if not raw_parent:
        return None
    s = raw_parent.strip().upper().replace("—", "-").replace("–", "-")
    if "-" not in s:
        return None
    if _V_DECIMAL_IN_RANGE_RE.match(s) and "." in s:
        return "icd9_span"
    if _V_PURE_RANGE_RE.match(s) and "." not in s:
        return "icd10_span"
    return None


def _trace_parent_span_hint(
    rec: dict,
    rows_by_norm: Dict[str, List[dict]],
    max_levels: int = 3,
) -> Optional[str]:
    """
    从当前记录起，沿同体系向上追溯最多 max_levels 层，尝试识别父范围形态。
    命中则返回 'ICD10CM' 或 'ICD9CM'；否则返回 None。
    """
    if not rec:
        return None
    sys_sel = rec.get("system", "")
    cur = rec
    levels = 0
    seen = set()
    while cur and levels < max_levels:
        raw_parent = (cur.get("parent_code") or "").strip()
        hint = _classify_v_parent_span(raw_parent)
        if hint == "icd10_span":
            return "ICD10CM"
        if hint == "icd9_span":
            return "ICD9CM"

        parent_norm = norm_code(cur.get("parent_code_norm", ""))
        if not parent_norm or parent_norm in seen:
            break
        seen.add(parent_norm)
        same_sys = [r for r in rows_by_norm.get(parent_norm, []) if r.get("system") == sys_sel]
        if not same_sys:
            break
        cur = same_sys[0]
        levels += 1
    return None


def infer_prefer_system_ev_aware(
    code_raw: Optional[str],
    code_norm: str,
    prefer_modern: bool = True,
) -> str:
    """
    EV-aware 的体系推断：
      - 数字开头 → ICD9CM
      - 字母开头且非 E/V → ICD10CM
      - V 码/E 码根据位置与范围启发式判断
    """
    cn = (code_norm or "").upper()
    cr = (code_raw or "").upper()

    if re.match(r"^\d", cn):
        return "ICD9CM"
    if re.match(r"^[A-Z]", cn) and not cn.startswith("E") and not cn.startswith("V"):
        return "ICD10CM"

    # V-codes
    if cn.startswith("V"):
        if re.search(r"[A-Z]", cn[1:]) or re.search(r"[A-Z]", cr[1:]):
            return "ICD10CM"
        return "ICD10CM" if prefer_modern else "ICD9CM"

    # E-codes
    if cn.startswith("E"):
        mraw = re.match(r"^E(\d+)", cr)
        if mraw:
            pre = cr.split(".")[0][1:]
            if len(pre) == 2:
                return "ICD10CM"
            if len(pre) >= 3 and pre[:3].isdigit() and int(pre[:3]) >= 800:
                return "ICD9CM"
        m = re.match(r"^E(\d+)", cn)
        if m:
            num = m.group(1)
            if len(num) == 2:
                return "ICD10CM"
            if len(num) >= 3 and int(num[:3]) >= 800:
                return "ICD10CM" if prefer_modern else "ICD9CM"
        return "ICD10CM"

    return "ICD10CM"


def _choose_with_v_parent_span(
    cands: List[dict],
    rows_by_norm: Dict[str, List[dict]],
    prefer_modern: bool,
) -> Tuple[Optional[dict], str]:
    """
    针对 V 码冲突，尝试通过父范围形态裁决。
    返回 (chosen_record_or_None, reason_suffix)
    """
    sys_group: Dict[str, dict] = {}
    for r in cands:
        s = r.get("system", "")
        if s in ("ICD9CM", "ICD10CM") and s not in sys_group:
            sys_group[s] = r
    r9 = sys_group.get("ICD9CM")
    r10 = sys_group.get("ICD10CM")

    if not r9 and not r10:
        return None, "v_parent_span=none"

    hint9 = _trace_parent_span_hint(r9, rows_by_norm) if r9 else None
    hint10 = _trace_parent_span_hint(r10, rows_by_norm) if r10 else None

    if hint9 == "ICD9CM" and hint10 != "ICD10CM":
        return r9, "v_parent_span=icd9"
    if hint10 == "ICD10CM" and hint9 != "ICD9CM":
        return r10, "v_parent_span=icd10"

    return None, "v_parent_span=none"


def select_record_smart(
    code_raw: str,
    code_norm: str,
    rows_by_norm: Dict[str, List[dict]],
    rec_by_code: Dict[str, List[dict]],
    amb: Optional[AmbiguityLogger] = None,
    prefer_modern: bool = True,
) -> Optional[dict]:
    """
    选择起点记录（叶子），并在歧义时记录选择过程：
      1) 优先 raw-code 精确匹配；
      2) 否则回落到 norm 桶；
      3) 再通过 V-parent-span + EV-aware 规则决策体系。
    """
    cr = (code_raw or "").strip().upper()
    cn = (code_norm or "").strip().upper()

    # 1) raw-code 精确匹配（最优）
    if cr and cr in rec_by_code:
        cand = rec_by_code[cr]
        if len(cand) == 1:
            return cand[0]
        chosen, reason_suffix = (None, "v_parent_span=skip")
        if cn.startswith("V"):
            chosen, reason_suffix = _choose_with_v_parent_span(
                cand, rows_by_norm, prefer_modern
            )
            if chosen is None:
                sys_chosen = infer_prefer_system_ev_aware(cr, cn, prefer_modern=prefer_modern)
                chosen = next(
                    (r for r in cand if r.get("system") == sys_chosen),
                    cand[0],
                )
                reason_suffix += "|ev_fallback"
        else:
            sys_chosen = infer_prefer_system_ev_aware(cr, cn, prefer_modern=prefer_modern)
            chosen = next(
                (r for r in cand if r.get("system") == sys_chosen),
                cand[0],
            )

        if amb is not None:
            amb.log(
                code_raw=cr,
                code_norm=cn,
                chosen_system=chosen.get("system", ""),
                alternatives=cand,
                reason="raw_code_multiple|" + reason_suffix,
            )
        return chosen

    # 2) fallback 到 norm 桶
    cand = rows_by_norm.get(cn, [])
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0]

    # 3) 多条 -> 先用 V-parent-span，再用 EV-aware
    chosen, reason_suffix = (None, "v_parent_span=skip")
    if cn.startswith("V"):
        chosen, reason_suffix = _choose_with_v_parent_span(
            cand, rows_by_norm, prefer_modern
        )
        if chosen is None:
            sys_chosen = infer_prefer_system_ev_aware(cr, cn, prefer_modern=prefer_modern)
            chosen = next(
                (r for r in cand if r.get("system") == sys_chosen),
                cand[0],
            )
            reason_suffix += "|ev_fallback"
    else:
        sys_chosen = infer_prefer_system_ev_aware(cr, cn, prefer_modern=prefer_modern)
        chosen = next(
            (r for r in cand if r.get("system") == sys_chosen),
            cand[0],
        )

    if amb is not None:
        amb.log(
            code_raw=cr,
            code_norm=cn,
            chosen_system=chosen.get("system", ""),
            alternatives=cand,
            reason="norm_multiple|" + reason_suffix,
        )
    return chosen


def _find_start_with_trim_anchor(
    code_raw: str,
    code_norm: str,
    rows_by_norm: Dict[str, List[dict]],
    rec_by_code: Dict[str, List[dict]],
    amb: Optional[AmbiguityLogger],
    prefer_modern: bool = True,
    max_trim: int = 10,
) -> Optional[dict]:
    """
    当精确匹配不到叶子记录时，按规范逐步截短 code_norm 的末尾字符，
    寻找最近存在的祖先节点作为路径起点。
    例如：H4011X0 -> H4011X -> H4011 -> H401 -> H40 ...
    """
    cn = (code_norm or "").strip().upper()
    cr = (code_raw or "").strip().upper()
    if not cn:
        return None

    trimmed = cn
    steps = 0
    while steps < max_trim and len(trimmed) > 1:
        trimmed = trimmed[:-1]
        steps += 1
        cands = rows_by_norm.get(trimmed, [])
        if not cands:
            continue
        sys_sel = infer_prefer_system_ev_aware(cr, trimmed, prefer_modern=prefer_modern)
        chosen = next((r for r in cands if r.get("system") == sys_sel), cands[0])
        if amb is not None:
            amb.log(
                code_raw=cr,
                code_norm=cn,
                chosen_system=chosen.get("system", ""),
                alternatives=cands,
                reason=f"fallback_trim_anchor|from={cn}|to={trimmed}",
            )
        return chosen
    return None


def get_name_for_code(
    code_norm: str,
    system: str,
    rows_by_norm: Dict[str, List[dict]],
) -> str:
    cand = rows_by_norm.get(code_norm, [])
    for r in cand:
        if r.get("system") == system:
            nm = r.get("name") or r.get("code") or code_norm
            return nm
    if cand:
        return cand[0].get("name") or cand[0].get("code") or code_norm
    return code_norm


def build_path_hardlock(
    code_raw: str,
    code_norm: str,
    rows_by_norm: Dict[str, List[dict]],
    rec_by_code: Dict[str, List[dict]],
    amb: Optional[AmbiguityLogger] = None,
    prefer_modern: bool = True,
    max_depth: int = 50,
) -> List[Tuple[str, str]]:
    """
    构造从 root ... 到 leaf 的路径（含两端），并**硬锁系统**：
      - 起点：select_record_smart
      - 向上：每一层都必须在同一 system 下找到父节点；找不到就停止（不跨系统）
      - 返回 [(code_norm, system), ...] 以 root->...->leaf 顺序
    """
    cn = (code_norm or "").strip().upper()
    if not cn:
        return []

    start = select_record_smart(
        code_raw, cn, rows_by_norm, rec_by_code, amb=amb, prefer_modern=prefer_modern
    )
    if start is None:
        start = _find_start_with_trim_anchor(
            code_raw,
            cn,
            rows_by_norm,
            rec_by_code,
            amb=amb,
            prefer_modern=prefer_modern,
        )
        if start is None:
            sys_sel = infer_prefer_system_ev_aware(code_raw, cn, prefer_modern=prefer_modern)
            return [(cn, sys_sel)]

    sys_sel = start["system"]

    seen = set()
    parents: List[Tuple[str, str]] = []
    parent = norm_code(start.get("parent_code_norm", ""))

    depth = 0
    while parent and parent not in seen and depth < max_depth:
        seen.add(parent)
        same_sys = [r for r in rows_by_norm.get(parent, []) if r.get("system") == sys_sel]
        if not same_sys:
            if amb is not None:
                amb.log(
                    code_raw="",
                    code_norm=parent,
                    chosen_system=sys_sel,
                    alternatives=rows_by_norm.get(parent, []),
                    reason="parent_missing_same_system",
                )
            break
        if len(same_sys) > 1 and amb is not None:
            amb.log(
                code_raw="",
                code_norm=parent,
                chosen_system=sys_sel,
                alternatives=same_sys,
                reason="parent_multiple_same_system",
            )
        parents.append((parent, sys_sel))
        parent = norm_code(same_sys[0].get("parent_code_norm", ""))
        depth += 1

    path = list(reversed(parents)) + [(cn, sys_sel)]
    return path


def take_labels_aligned_by_system(
    path: List[Tuple[str, str]],
    rows_by_norm: Dict[str, List[dict]],
) -> List[str]:
    """
    颗粒度对齐策略：
      - ICD9CM: 取路径第 2–5 层
      - ICD10CM: 取路径第 1–4 层
    若路径不足，对应层重复最后一个有效层名称。
    始终返回长度为 4 的标签列表。
    """
    if not path:
        return [PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
    leaf_sys = path[-1][1] if path and len(path[-1]) >= 2 else ""
    start_idx = 0 if leaf_sys == "ICD10CM" else 1
    labels: List[str] = []
    for i in range(4):
        idx = start_idx + i
        if idx >= len(path):
            idx = len(path) - 1
        cn_i, sys_i = path[idx]
        labels.append(get_name_for_code(cn_i, sys_i, rows_by_norm))
    return labels


# =============================
# CLI & 主流程
# =============================


def parse_args() -> argparse.Namespace:
    parser = make_cli_parser()
    # 复用 --ontology-csv 作为合并本体表路径
    parser.add_argument(
        "--max-depth",
        type=int,
        default=50,
        help="Maximum parent traversal depth when building hierarchies",
    )
    parser.add_argument(
        "--prefer-modern",
        dest="prefer_modern",
        action="store_true",
        help="Ambiguous V/E codes prefer ICD-10 (modern) if True.",
    )
    parser.add_argument(
        "--no-prefer-modern",
        dest="prefer_modern",
        action="store_false",
        help="Ambiguous V/E codes prefer ICD-9 if set.",
    )
    parser.set_defaults(prefer_modern=True)
    return parser.parse_args()


def maybe_dump_tag_counts(vocabs: Sequence[Sequence[str]], out_dir: str) -> None:
    counts = [len(v) for v in vocabs]
    path = os.path.join(out_dir, "tag_class_counts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(counts, f)


def main() -> None:
    args = parse_args()

    # 1. 从 mimic-iv_data/voc_final.pkl 中读取诊断代码顺序（diag_voc）
    codes = load_voc_codes(args.voc_path, "diag_voc")

    # 2. 加载合并本体（ICD9CM + ICD10CM）
    rows_by_norm, rec_by_code = load_combined_table(args.ontology_csv)

    # 3. 为每个 code 构造路径与 4 层标签
    os.makedirs(args.out_dir, exist_ok=True)
    vocab_order_path = os.path.join(args.out_dir, "icd_vocab_order.txt")
    write_vocab_order(codes, vocab_order_path)

    amb = AmbiguityLogger()

    items_path = os.path.join(args.out_dir, "icd_items.jsonl")
    tag_sequences: List[List[str]] = []

    with open(items_path, "w", encoding="utf-8") as fw:
        for code_raw in codes:
            cn = norm_code(code_raw)
            path = build_path_hardlock(
                code_raw=str(code_raw),
                code_norm=cn,
                rows_by_norm=rows_by_norm,
                rec_by_code=rec_by_code,
                amb=amb,
                prefer_modern=args.prefer_modern,
                max_depth=args.max_depth,
            )
            labels = take_labels_aligned_by_system(path, rows_by_norm)
            tag_sequences.append(labels)

            # 叶子文本作为 item 文本
            leaf_cn, leaf_sys = path[-1] if path else (cn, infer_prefer_system_ev_aware(code_raw, cn))
            leaf_text = get_name_for_code(leaf_cn, leaf_sys, rows_by_norm)

            rec = {
                "code": str(code_raw),
                "text": leaf_text,
                "tags": labels,
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4. 构建标签词表与索引矩阵
    vocabs = build_tag_vocabs(tag_sequences, LEVEL_LABELS, args.out_dir)
    maybe_dump_tag_counts(vocabs, args.out_dir)

    tag_idx = build_tag_indices_tensor(tag_sequences, vocabs)
    save_tensor(tag_idx, os.path.join(args.out_dir, "tags_indices.pt"))

    # 5. 可选：生成文本与标签的 BioBERT embeddings
    if args.embed:
        from hidvae_prep_utils import load_vocab_tokens, save_embeddings

        items = []
        with open(items_path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        texts = [rec["text"] if rec["text"] else rec["code"] for rec in items]
        save_embeddings(
            texts,
            os.path.join(args.out_dir, "embeddings", "icd_text_emb.pt"),
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

    # 6. 歧义记录导出
    amb_path = os.path.join(args.out_dir, "ambiguous_codes.csv")
    amb.dump_csv(amb_path)


if __name__ == "__main__":
    main()


