import os
import torch
from config import config
from utils import seed_everything, dill_load
from trainer import GRU4CMRTrainer

def run_single_model(config):
    perf_test = bool(config.get('PERF_TEST', False))
    if not perf_test:
        print("the downstream model is gru")
        print(config)

    gpu_only = bool(config.get('GPU_ONLY', False))
    want_cuda = bool(config.get('USE_CUDA', False)) or gpu_only
    if want_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU execution, but no CUDA GPUs are available.")

        gpu_index = str(config.get('GPU', '0'))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
        device = torch.device('cuda:0')
        if not perf_test:
            print(device)
            print(torch.tensor(0).to(device))
            print(os.environ['CUDA_VISIBLE_DEVICES'])
        torch.cuda.set_device(device)
        if not perf_test:
            print(device)
    else:
        device = torch.device('cpu')
        if not perf_test:
            print(device)

    root = "mimic-iii_data/"
    data_path = config['ROOT'] + root + "records_final.pkl"
    voc_path = config['ROOT'] + root + "voc_final.pkl"
    ddi_adj_path = config['ROOT'] + root + "ddi_A_final.pkl"

    ddi_adj = dill_load(ddi_adj_path)
    data = dill_load(data_path)
    voc = dill_load(voc_path)

    #两种模式: semantic id or semantic embedding
    hidvae_embeddings = None
    hidvae_sids = None
    sid_aggregation = config.get('SID_AGGREGATION', 'sum')
    if config.get('HIDVAE_USE_SID', False):
        hidvae_diag_sid = torch.load(os.path.join(config['SEMANTIC_ROOT'], 'icd', 'hidvae_sid.pt') if config['HIDVAE_DIAG_SID'] is None else config['HIDVAE_DIAG_SID'], map_location='cpu').long()
        hidvae_proc_sid = torch.load(os.path.join(config['SEMANTIC_ROOT'], 'proc', 'hidvae_sid.pt') if config['HIDVAE_PROC_SID'] is None else config['HIDVAE_PROC_SID'], map_location='cpu').long()
        hidvae_drug_sid = torch.load(os.path.join(config['SEMANTIC_ROOT'], 'drug', 'hidvae_sid.pt') if config['HIDVAE_DRUG_SID'] is None else config['HIDVAE_DRUG_SID'], map_location='cpu').long()
        hidvae_sids = (hidvae_diag_sid, hidvae_proc_sid, hidvae_drug_sid)
    else:
        hidvae_diag_emb = torch.load(config['HIDVAE_DIAG_EMB'], map_location='cpu').float()
        hidvae_proc_emb = torch.load(config['HIDVAE_PROC_EMB'], map_location='cpu').float()
        hidvae_drug_emb = torch.load(config['HIDVAE_DRUG_EMB'], map_location='cpu').float()
        hidvae_embeddings = (hidvae_diag_emb, hidvae_proc_emb, hidvae_drug_emb)

    trainer = GRU4CMRTrainer(
        config,
        device,
        (ddi_adj, data, voc),
        medrq_embeddings=hidvae_embeddings,
        medrq_sids=hidvae_sids,
        sid_aggregation=sid_aggregation,
    )

    if perf_test:
        elapsed = trainer.perf_forward_all(
            threshold=float(config.get('PERF_TEST_THRESHOLD', 0.5) or 0.5),
            warmup_rounds=int(config.get('PERF_TEST_WARMUP_ROUNDS', 1) or 0),
        )
        print(f"{elapsed:.6f}")
        return
    else:
        trainer.main()

    print("Everything is OK!")


if __name__ == '__main__':
    config = config
    seed_everything(config['SEED'])
    run_single_model(config)
