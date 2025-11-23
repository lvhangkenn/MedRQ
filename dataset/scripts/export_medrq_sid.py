import argparse
import os
import sys
from typing import Literal

import torch


MODALITY_CONFIG = {
    "icd": {
        "text_emb": os.path.join("embeddings", "icd_text_emb.pt"),
        "order_file": "icd_vocab_order.txt",
    },
    "proc": {
        "text_emb": os.path.join("embeddings", "proc_text_emb.pt"),
        "order_file": "proc_vocab_order.txt",
    },
    "drug": {
        "text_emb": os.path.join("embeddings", "drug_text_emb.pt"),
        "order_file": "drug_vocab_order.txt",
    },
}


def _ensure_project_root_on_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_hrqvae_from_checkpoint(ckpt_path: str, device: torch.device):
    _ensure_project_root_on_path()
    from modules.h_rqvae import HRqVae  # noqa: WPS433

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("model_config", {}) or {}

    model = HRqVae(
        input_dim=cfg.get("input_dim", 768),
        embed_dim=cfg.get("embed_dim", 64),
        hidden_dims=cfg.get("hidden_dims", [512, 256]),
        codebook_size=cfg.get("codebook_size", 512),
        codebook_kmeans_init=False,
        codebook_normalize=cfg.get("codebook_normalize", True),
        codebook_sim_vq=cfg.get("codebook_sim_vq", False),
        codebook_mode=cfg.get("codebook_mode"),
        n_layers=cfg.get("n_layers", 4),
        commitment_weight=cfg.get("commitment_weight", 0.25),
        n_cat_features=cfg.get("n_cat_features", 0),
        tag_alignment_weight=cfg.get("tag_alignment_weight", 0.5),
        tag_prediction_weight=cfg.get("tag_prediction_weight", 0.5),
        tag_class_counts=cfg.get("tag_class_counts"),
        tag_embed_dim=cfg.get("tag_embed_dim", 768),
        use_focal_loss=cfg.get("use_focal_loss", False),
        focal_loss_params=cfg.get("focal_loss_params"),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        use_batch_norm=cfg.get("use_batch_norm", True),
        alignment_temperature=cfg.get("alignment_temperature", 0.1),
        sem_id_uniqueness_weight=cfg.get("sem_id_uniqueness_weight", 0.5),
        sem_id_uniqueness_margin=cfg.get("sem_id_uniqueness_margin", 0.5),
    ).to(device)

    # Load weights (be tolerant to minor mismatches)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def export_sids(
    ckpt_path: str,
    modality: Literal["icd", "proc", "drug"],
    semantic_dir: str,
    out_path: str,
    batch_size: int = 256,
    device_str: str | None = None,
) -> None:
 
    if modality not in MODALITY_CONFIG:
        raise ValueError(f"Unsupported modality '{modality}'. Expected one of {list(MODALITY_CONFIG)}")

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    semantic_dir = os.path.abspath(semantic_dir)
    out_path = os.path.abspath(out_path)

    model, cfg = load_hrqvae_from_checkpoint(ckpt_path, device)
    n_layers = int(cfg.get("n_layers", 4))

    modal_cfg = MODALITY_CONFIG[modality]
    text_emb_path = os.path.join(semantic_dir, modal_cfg["text_emb"])
    order_path = os.path.join(semantic_dir, modal_cfg["order_file"])

    if not os.path.exists(text_emb_path):
        raise FileNotFoundError(f"Missing text embedding file: {text_emb_path}")
    if not os.path.exists(order_path):
        raise FileNotFoundError(f"Missing vocab order file: {order_path}")

    x_all = torch.load(text_emb_path, map_location="cpu", weights_only=False).float()
    num_items = x_all.shape[0]


    sid_chunks: list[torch.Tensor] = []
    for start in range(0, num_items, batch_size):
        batch = x_all[start:start + batch_size].to(device)
        encoded = model.encode(batch)
        out = model.get_semantic_ids(encoded)
        sid_chunks.append(out.sem_ids.transpose(0, 1).to("cpu", dtype=torch.long))

    sids = torch.cat(sid_chunks, dim=0)  # [num_items, n_layers]
    assert sids.shape[0] == num_items and sids.shape[1] == n_layers, f"Unexpected SID shape: {sids.shape}, expected [{num_items}, {n_layers}]"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(sids, out_path)

    print(f"Exported {modality} semantic IDs: {sids.shape} -> {out_path}")
    print(f"Aligned to vocab order: {order_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HRqVAE discrete semantic IDs (SIDs) for medical modalities.")
    parser.add_argument("--checkpoint_path", type=str, default="/root/vae药物推荐_未消融/HiD-VAE/out_有碰撞_kmeans/hrqvae/medical/hrqvae_MEDICAL_MIMIC3_20251106_162829/hrqvae_checkpoint_iter80000.pt", help="Path to HRqVAE checkpoint.")
    parser.add_argument("--modality", type=str, choices=list(MODALITY_CONFIG), default="icd", help="Target modality: icd|proc|drug.")
    parser.add_argument("--semantic_dir", type=str, default="/root/vae药物推荐_未消融/MedAlign/semantic/drug", help="Directory containing semantic data (e.g. MedAlign/semantic/icd).")
    parser.add_argument("--out_path", type=str, default="/root/vae药物推荐_未消融/MedAlign/semantic/drug/medrq_sid.pt", help="Output .pt file path for SIDs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device identifier, default auto.")
    args = parser.parse_args()

    export_sids(
        ckpt_path=args.checkpoint_path,
        modality=args.modality,
        semantic_dir=args.semantic_dir,
        out_path=args.out_path,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from typing import Literal

import torch


MODALITY_CONFIG = {
    "icd": {
        "text_emb": os.path.join("embeddings", "icd_text_emb.pt"),
        "order_file": "icd_vocab_order.txt",
    },
    "proc": {
        "text_emb": os.path.join("embeddings", "proc_text_emb.pt"),
        "order_file": "proc_vocab_order.txt",
    },
    "drug": {
        "text_emb": os.path.join("embeddings", "drug_text_emb.pt"),
        "order_file": "drug_vocab_order.txt",
    },
}


def _ensure_project_root_on_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_hrqvae_from_checkpoint(ckpt_path: str, device: torch.device):
    """
    与 export_hrqvae_embeddings.py 保持一致的 checkpoint 加载逻辑。
    """
    _ensure_project_root_on_path()
    from modules.h_rqvae import HRqVae  # noqa: WPS433

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("model_config", {}) or {}

    model = HRqVae(
        input_dim=cfg.get("input_dim", 768),
        embed_dim=cfg.get("embed_dim", 64),
        hidden_dims=cfg.get("hidden_dims", [512, 256]),
        codebook_size=cfg.get("codebook_size", 512),
        codebook_kmeans_init=False,
        codebook_normalize=cfg.get("codebook_normalize", True),
        codebook_sim_vq=cfg.get("codebook_sim_vq", False),
        codebook_mode=cfg.get("codebook_mode"),
        n_layers=cfg.get("n_layers", 4),
        commitment_weight=cfg.get("commitment_weight", 0.25),
        n_cat_features=cfg.get("n_cat_features", 0),
        tag_alignment_weight=cfg.get("tag_alignment_weight", 0.5),
        tag_prediction_weight=cfg.get("tag_prediction_weight", 0.5),
        tag_class_counts=cfg.get("tag_class_counts"),
        tag_embed_dim=cfg.get("tag_embed_dim", 768),
        use_focal_loss=cfg.get("use_focal_loss", False),
        focal_loss_params=cfg.get("focal_loss_params"),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        use_batch_norm=cfg.get("use_batch_norm", True),
        alignment_temperature=cfg.get("alignment_temperature", 0.1),
        sem_id_uniqueness_weight=cfg.get("sem_id_uniqueness_weight", 0.5),
        sem_id_uniqueness_margin=cfg.get("sem_id_uniqueness_margin", 0.5),
    ).to(device)

    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def export_sids(
    ckpt_path: str,
    modality: Literal["icd", "proc", "drug"],
    semantic_dir: str,
    out_path: str,
    layout: Literal["semantic_only", "concat", "interleaved"] = "semantic_only",
    batch_size: int = 256,
    device_str: str | None = None,
) -> None:

    if modality not in MODALITY_CONFIG:
        raise ValueError(f"Unsupported modality '{modality}'. Expected one of {list(MODALITY_CONFIG)}")

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    semantic_dir = os.path.abspath(semantic_dir)
    out_path = os.path.abspath(out_path)

    model, _ = load_hrqvae_from_checkpoint(ckpt_path, device)

    modal_cfg = MODALITY_CONFIG[modality]
    text_emb_path = os.path.join(semantic_dir, modal_cfg["text_emb"])
    order_path = os.path.join(semantic_dir, modal_cfg["order_file"])

    if not os.path.exists(text_emb_path):
        raise FileNotFoundError(f"Missing text embedding file: {text_emb_path}")
    if not os.path.exists(order_path):
        raise FileNotFoundError(f"Missing vocab order file: {order_path}")

    x_all = torch.load(text_emb_path, map_location="cpu", weights_only=False).float()
    num_items = x_all.shape[0]

    sid_chunks: list[torch.Tensor] = []

    for start in range(0, num_items, batch_size):
        batch = x_all[start:start + batch_size].to(device)
        encoded = model.encode(batch)  # [B, embed_dim]
        out = model.get_semantic_ids(encoded)  # HRqVaeOutput, sem_ids: [n_layers, B]
        sem_ids = out.sem_ids.transpose(0, 1).to(torch.long)  # [B, n_layers]

        if layout == "semantic_only":
            sids_out = sem_ids
        else:
            tag_pred = model.predict_tags(batch)
            tag_ids = tag_pred["predictions"].to(torch.long)  # [B, n_layers_tag]

            if tag_ids.dim() == 3:
                tag_ids = tag_ids.squeeze(1)

            if layout == "concat":
                sids_out = torch.cat([sem_ids, tag_ids], dim=1)  # [B, n_layers + n_tag_layers]
            elif layout == "interleaved":
                # 交错 [s1, t1, s2, t2, ...], 参考medrq
                bsz, n_sem = sem_ids.shape
                n_tag = tag_ids.shape[1]
                max_len = max(n_sem, n_tag)
                parts = []
                for i in range(max_len):
                    if i < n_sem:
                        parts.append(sem_ids[:, i:i+1])
                    if i < n_tag:
                        parts.append(tag_ids[:, i:i+1])
                sids_out = torch.cat(parts, dim=1)
            else:
                raise ValueError(f"Unsupported layout '{layout}'")

        sid_chunks.append(sids_out.cpu())

    sids = torch.cat(sid_chunks, dim=0)  # [N, D]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(sids, out_path)

    print(f"Exported {modality} SIDs ({layout}): {sids.shape} -> {out_path}")
    print(f"Aligned to vocab order: {order_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HRqVAE discrete SIDs for medical modalities.")
    parser.add_argument("--checkpoint_path", type=str, default="/root/vae药物推荐_未消融/HiD-VAE/out_mimic3_icd_有碰撞/hrqvae/medical/hrqvae_MEDICAL_MIMIC3_20251106_161312/hrqvae_checkpoint_iter95000.pt", help="Path to HRqVAE checkpoint.")
    parser.add_argument("--modality", type=str, choices=list(MODALITY_CONFIG), default="icd", help="Target modality.")
    parser.add_argument("--semantic_dir", type=str, default="/root/vae药物推荐_未消融/MedAlign/semantic/icd" , help="Directory with semantic data (e.g. MedAlign/semantic/icd).")
    parser.add_argument("--out_path", type=str, default="/root/vae药物推荐_未消融/MedAlign/semantic/icd/medrq_sid.pt", help="Output .pt path for exported SIDs.")
    parser.add_argument("--layout", type=str, choices=["semantic_only", "concat", "interleaved"], default="semantic_only", help="How to organize IDs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device identifier, default auto.")
    args = parser.parse_args()

    export_sids(
        ckpt_path=args.checkpoint_path,
        modality=args.modality,
        semantic_dir=args.semantic_dir,
        out_path=args.out_path,
        layout=args.layout,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()


