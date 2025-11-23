import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class LayeredSIDEmbedding(nn.Module):
 
    def __init__(self, sids: torch.Tensor, emb_dim: int, aggregation: str = 'sum', device: torch.device | None = None):
        super().__init__()
        assert sids.dim() == 2, "sids shape must be [num_items, n_layers]"
        self.register_buffer('item_layer_ids', sids.long())  # [N, L]
        self.num_items, self.n_layers = sids.shape
        self.emb_dim = emb_dim
        self.aggregation = aggregation
        # 估计 codebook 大小（每层词表大小）为该模态下SID的最大值+1
        self.num_embeddings = int(sids.max().item()) + 1 if sids.numel() > 0 else 1
        self.total_embeddings = self.num_embeddings * self.n_layers
        self.emb_table = nn.Embedding(self.total_embeddings, emb_dim)
        if device is not None:
            self.to(device)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        original_shape = ids.shape
        layer_ids = self.item_layer_ids[ids]  # [M, L]
        offsets = torch.arange(self.n_layers, device=layer_ids.device).view(1, -1) * self.num_embeddings
        flat_indices = (layer_ids + offsets).view(-1)  # [M*L]
        emb = self.emb_table(flat_indices).view(-1, self.n_layers, self.emb_dim)  # [M, L, D]
        if self.aggregation == 'sum':
            out = emb.sum(dim=1)  # [M, D]
        elif self.aggregation == 'concat':
            out = emb.reshape(emb.shape[0], self.n_layers * self.emb_dim)  # [M, L*D]
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        last_dim = self.emb_dim if self.aggregation == 'sum' else self.n_layers * self.emb_dim
        return out.view(*original_shape, last_dim)

class MLP_Decoder(nn.Module):
    def __init__(self, hdim, nclass):
        super(MLP_Decoder, self).__init__()
        self.final_layer = nn.Linear(hdim, nclass)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        output = self.final_layer(h)
        return output

class Drug_MLP(nn.Module):
    def __init__(self, drug_dim, n_class):
        super(Drug_MLP, self).__init__()
        self.decoder = MLP_Decoder(drug_dim, n_class)

    def forward(self, drug_dim):
        pred = self.decoder(drug_dim)
        return pred

class GRU4CMR(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        ddi_matrix,
        device,
        medrq_embeddings=None,
        medrq_sids=None,
        sid_aggregation: str = 'sum',
    ):
        super(GRU4CMR, self).__init__()
        # vocab_size: (n_diag, n_proc, n_med)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device
        self.use_medrq_embeddings = medrq_embeddings is not None
        self.use_medrq_sid = medrq_sids is not None
        self.sid_aggregation = sid_aggregation

        # 构建三类嵌入
        if self.use_medrq_embeddings:
            diag_emb, proc_emb, drug_emb = medrq_embeddings

            if diag_emb.shape[1] != emb_dim or proc_emb.shape[1] != emb_dim or drug_emb.shape[1] != emb_dim:
                raise ValueError("HiD-VAE embedding dimensionality must match GRU4CMR embedding dimension.")

            self.diagnose_emb = nn.Embedding.from_pretrained(diag_emb.clone().detach().float(), freeze=False)
            self.procedure_emb = nn.Embedding.from_pretrained(proc_emb.clone().detach().float(), freeze=False)
            self.drug_emb = nn.Embedding.from_pretrained(drug_emb.clone().detach().float(), freeze=False)
        elif self.use_medrq_sid:
            diag_sid, proc_sid, drug_sid = medrq_sids
            self.diagnose_emb = LayeredSIDEmbedding(diag_sid, emb_dim, aggregation=self.sid_aggregation, device=self.device)
            self.procedure_emb = LayeredSIDEmbedding(proc_sid, emb_dim, aggregation=self.sid_aggregation, device=self.device)
            self.drug_emb = LayeredSIDEmbedding(drug_sid, emb_dim, aggregation=self.sid_aggregation, device=self.device)
        else:
            self.diagnose_emb = nn.Embedding(vocab_size[0], emb_dim)
            self.procedure_emb = nn.Embedding(vocab_size[1], emb_dim)
            self.drug_emb = nn.Embedding(vocab_size[2], emb_dim)

        if self.use_medrq_sid:
            diag_in_dim = emb_dim if self.sid_aggregation == 'sum' else self.diagnose_emb.n_layers * emb_dim
            proc_in_dim = emb_dim if self.sid_aggregation == 'sum' else self.procedure_emb.n_layers * emb_dim
            drug_in_dim = emb_dim if self.sid_aggregation == 'sum' else self.drug_emb.n_layers * emb_dim
        else:
            diag_in_dim = emb_dim
            proc_in_dim = emb_dim
            drug_in_dim = emb_dim
        self.diag_in_dim = diag_in_dim
        self.proc_in_dim = proc_in_dim
        self.drug_in_dim = drug_in_dim

        self.dropout = nn.Dropout(p=0.7, inplace=False)
        self.diagnose_encoder = nn.GRU(diag_in_dim, diag_in_dim, batch_first=True)
        self.procedure_encoder = nn.GRU(proc_in_dim, proc_in_dim, batch_first=True)
        self.his_encoder = nn.GRU(drug_in_dim, drug_in_dim, batch_first=True)

        self.init_weights()

        self.combined_dim = self.diag_in_dim + self.proc_in_dim + self.drug_in_dim
        self.drug_mlp = Drug_MLP(self.combined_dim, self.vocab_size[2]).to(self.device)
        self.ddi_matrix = ddi_matrix

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

        self.apply(_init_weights)

    def forward(self, input):
        """
        :param input: patient health [[[],[],[]], [[],[],[]]]
        :param drug_his: patient drug history

        :return: prob, ddi rate
        """
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        # patient health representation
        diag_seq = []
        proc_seq = []
        his_med_seq = []

        history_meds = []

        for adm in input:
            if history_meds:
                ids_tensor = torch.LongTensor(history_meds).unsqueeze(0).to(self.device)
                his_drug_emb_block = self.drug_emb(ids_tensor)  # [1, len, D_out(drug)]
                his_med_emb = sum_embedding(self.dropout(his_drug_emb_block))
            else:
                his_med_emb = torch.zeros(1, 1, self.drug_in_dim).to(self.device)

            his_med_seq.append(his_med_emb)

            diag = sum_embedding(self.dropout(self.diagnose_emb(torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))))  # (1,1,D_in_diag or emb_dim)
            proc = sum_embedding(self.dropout(self.procedure_emb(torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))))

            diag_seq.append(diag)
            proc_seq.append(proc)
            history_meds.extend(adm[-1])

        # 拼接成 Tensor
        diag_seq = torch.cat(diag_seq, dim=1)  # (1,seq,dim)
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,seq,dim)
        his_med_seq = torch.cat(his_med_seq, dim=1) if his_med_seq else torch.zeros(1, 1, self.drug_in_dim).to(self.device)

        o1, h1 = self.diagnose_encoder(diag_seq)  # o: all，h: last
        o2, h2 = self.procedure_encoder(proc_seq)
        o3, h3 = self.his_encoder(his_med_seq)

        visit = torch.cat([o1, o2, o3], dim=-1).squeeze(dim=0)
        
        query = visit[-1:, :]
        
        result = self.drug_mlp(query)

        # calculate ddl
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        ddi_rate = 0.0005 * neg_pred_prob.mul(self.ddi_matrix).sum()

        return result, ddi_rate
        