import math
import dill
import torch
import os
import time
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from utils import train_test_split, calculate_metrics, ddi_rate_score, llprint
from model.models import GRU4CMR

class GRU4CMRTrainer():
    def __init__(self, config, device, datasets, medrq_embeddings=None, medrq_sids=None, sid_aggregation: str = 'sum'):
        super(GRU4CMRTrainer, self).__init__()
        self.use_cuda = config["USE_CUDA"]
        self.device = device
        # 仅保留核心数据：DDI 矩阵、就诊序列数据、词表
        self.ddi_adj, self.data, self.voc = datasets
        self.medrq_embeddings = medrq_embeddings
        self.medrq_sids = medrq_sids
        self.sid_aggregation = sid_aggregation
        if medrq_embeddings is not None:
            diag_emb, proc_emb, drug_emb = medrq_embeddings
            self.medrq_diag_emb = diag_emb.clone().detach().float()
            self.medrq_proc_emb = proc_emb.clone().detach().float()
            self.medrq_drug_emb = drug_emb.clone().detach().float()
        else:
            self.medrq_diag_emb = None
            self.medrq_proc_emb = None
            self.medrq_drug_emb = None

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                self.data[i][j] = self.data[i][j][:3]
        # print(self.data)
        self.root = config["ROOT"]
        self.task = config["TASK"]
        self.epochs = config["EPOCH"]
        self.embed_dim = config["DIM"]
        self.batch_sz = config["BATCH"]
        self.lr = config["LR"]
        self.wd = config["WD"]
        self.ratio = config["RATIO"]

        self.target_ddi = config["DDI"]
        self.kp = config['KP']
        self.model_name = config["MODEL"]
        self.log = config['LOG']
        self.hist = config['HIST']
        # few-shot configs
        self.few_shot = bool(config.get('FEWSHOT', False))
        self.few_shot_ratio = float(config.get('FEWSHOT_RATIO', 0.0) or 0.0)

        self.result = {}

        self.get_data()


    def get_data(self):
        # voc
        self.diag_voc, self.pro_voc, self.med_voc = self.voc["diag_voc"], self.voc["pro_voc"], self.voc["med_voc"]
        # 只需要三类词表：诊断、手术、用药
        self.voc_size = (
            len(self.diag_voc.idx2word),
            len(self.pro_voc.idx2word),
            len(self.med_voc.idx2word),
        )
        if self.medrq_embeddings is not None:
            if self.medrq_diag_emb.shape[0] != self.voc_size[0]:
                raise ValueError("HiD-VAE diagnosis embedding size does not match diag vocabulary.")
            if self.medrq_proc_emb.shape[0] != self.voc_size[1]:
                raise ValueError("HiD-VAE procedure embedding size does not match procedure vocabulary.")
            if self.medrq_drug_emb.shape[0] != self.voc_size[2]:
                raise ValueError("HiD-VAE medication embedding size does not match medication vocabulary.")
        if self.medrq_sids is not None:
            diag_sid, proc_sid, drug_sid = self.medrq_sids
            if diag_sid.shape[0] != self.voc_size[0]:
                raise ValueError("HiD-VAE diagnosis SID size does not match diag vocabulary.")
            if proc_sid.shape[0] != self.voc_size[1]:
                raise ValueError("HiD-VAE procedure SID size does not match procedure vocabulary.")
            if drug_sid.shape[0] != self.voc_size[2]:
                raise ValueError("HiD-VAE medication SID size does not match medication vocabulary.")

        self.ddi_adj = torch.tensor(self.ddi_adj, dtype=torch.float).to(self.device)

        split_point = int(len(self.data) * self.ratio)
        self.data_train, self.data_valid, self.data_test = train_test_split(self.data, split_point)

        # Few-shot: 下采样训练患者（只作用于训练集，避免泄露）
        if self.few_shot:
            if len(self.data_train) > 0 and 0.0 < self.few_shot_ratio < 1.0:
                rng = np.random.default_rng(seed=42)
                keep_n = max(1, int(round(len(self.data_train) * self.few_shot_ratio)))
                idxs = rng.choice(len(self.data_train), size=keep_n, replace=False)
                self.data_train = [self.data_train[i] for i in sorted(idxs.tolist())]
                print(f"[Few-Shot] 训练集下采样到 {keep_n}/{split_point} (~{self.few_shot_ratio*100:.1f}%) 位患者")
            else:
                print("[Few-Shot] 配置无效（比例范围应在 (0,1) 且训练集非空），跳过下采样")




    def get_model(self):
        model = GRU4CMR(
            self.voc_size,
            self.embed_dim,
            self.ddi_adj,
            self.device,
            medrq_embeddings=(
                self.medrq_diag_emb,
                self.medrq_proc_emb,
                self.medrq_drug_emb,
            )
            if self.medrq_embeddings is not None
            else None,
            medrq_sids=self.medrq_sids,
            sid_aggregation=self.sid_aggregation,
        )
        model = model.to(self.device)
        return model

    def get_opt(self, model):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def train(self, model, data, criterion, optimizer, epoch):
        print('Training Epoch {}:'.format(epoch))
        model.to(self.device)
        model.train()
        ddi_losses_epoch = []
        pre = [0.06 for i in range(self.hist)]
        pre_ddi = [0 for i in range(self.hist)]
        model.Modality_Alignment()
        for step, input in enumerate(data):  # patient
            loss = 0
            for idx, adm in enumerate(input):  # visit
                seq_input = input[: idx + 1]  # k visit
                # print(seq_input)
                loss_bce_target = np.zeros((1, self.voc_size[2]))
                loss_bce_target[:, adm[2]] = 1
                loss_multi_target = np.full((1, self.voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item
                # result, loss_ddi,ot_loss_structure,ot_loss_text = model(seq_input)

                result, loss_ddi = model(seq_input)
                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(self.device)
                )
                loss_multi = F.multilabel_margin_loss(
                    torch.sigmoid(result), torch.LongTensor(loss_multi_target).to(self.device)
                )
                result = torch.sigmoid(result).detach().cpu().numpy()[0]


                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]  # [1, drug_num]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], self.ddi_adj
                )

                # # stack
                pre.pop(0)
                pre.append(current_ddi_rate)
                avg_ddi = sum(pre) / len(pre)

                pre_ddi.pop(0)
                pre_ddi.append(loss_ddi)

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())

                if avg_ddi <= self.target_ddi:
                    if current_ddi_rate <= self.target_ddi:
                        loss = 0.95 * loss_bce + 0.05 * loss_multi
                    else:
                        loss = (
                                0.99 * (0.95 * loss_bce + 0.05 * loss_multi)
                                + 0.01 * loss_ddi  #
                        )
                else:
                    y = 0.95

                    beta = self.kp * (1 / (avg_ddi / self.target_ddi - 1 + 1e-8))
                    beta = min(math.tanh(beta), 1)
                    loss = (
                            beta * (y * loss_bce + (1 - y) * loss_multi)
                            + (1 - beta) * loss_ddi
                    )
                loss=loss.to(self.device)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1, norm_type=2)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data)))
        print()
        return sum(ddi_losses_epoch) / len(ddi_losses_epoch)

    def eval(self, model, data):
        print('Evaluating:')
        model.eval()
        smm_record = []
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0

        for step, input in enumerate(data):
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
            for adm_idx, adm in enumerate(input):
                target_output, *_ = model(input[: adm_idx + 1])
                y_gt_tmp = np.zeros(self.voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)  # binary label

                # prediction prod
                target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
                y_pred_prob.append(target_output)

                # prediction med set
                #原0.5
                y_pred_tmp = target_output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)

                # prediction label
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))

                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)
            smm_record.append(y_pred_label)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = calculate_metrics(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint("\rtest step: {} / {}".format(step, len(data)))

        # ddi rate
        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)

        llprint(
            "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                ddi_rate,
                np.mean(ja),
                np.mean(prauc),
                np.mean(avg_p),
                np.mean(avg_r),
                np.mean(avg_f1),
                med_cnt / visit_cnt,
            )
        )

        return (
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )

    def perf_forward_all(self, threshold: float = 0.5, warmup_rounds: int = 1) -> float:
        """
        在整个测试集上逐次做模型前向（概率化、阈值化、提取药物索引），仅计时该过程。
        - 不做指标计算、分组统计、文件写入、日志打印
        - 含不计时的预热阶段（warmup_rounds 次）
        返回：总耗时（秒）
        """
        model = self.get_model()
        model.eval()
        with torch.no_grad():
            # 对齐/准备（SID 模式会直接返回）
            model.Modality_Alignment()
            # 预热：不计时
            if warmup_rounds > 0 and len(self.data_test) > 0:
                sample = self.data_test[0]
                for _ in range(int(warmup_rounds)):
                    _ = model(sample[:1])
                if self.use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
            # 计时开始
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for input in self.data_test:
                for adm_idx, _ in enumerate(input):
                    logits, *_ = model(input[: adm_idx + 1])
                    prob = torch.sigmoid(logits)  # [1, n_class]
                    pred_mask = prob >= threshold
                    # 强制计算出索引，避免延迟求值
                    _ = torch.nonzero(pred_mask[0], as_tuple=False).view(-1)
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        return elapsed

    def main(self, test=False, case=False):
        model = self.get_model()
        optimizer = self.get_opt(model)
        criterion = None
        history = defaultdict(list)
        best_epoch, best_ja = 0, 0

        ddi_losses, ddi_values = [], []
        epochs_without_improvement = 0
        patience = 20

        for epoch in range(self.epochs):
            start = time.time()
            ddi_mean = self.train(model, self.data_train, criterion, optimizer, epoch)
            ddi_losses.append(ddi_mean)
            end = time.time()
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = self.eval(
                model, self.data_test)
            print(
                "all time: {}, test time: {}".format(
                    time.time() - start, time.time() - end
                )
            )
            ddi_values.append(ddi_rate)

            history["ja"].append(ja)
            history["ddi_rate"].append(ddi_rate)
            history["avg_p"].append(avg_p)
            history["avg_r"].append(avg_r)
            history["avg_f1"].append(avg_f1)
            history["prauc"].append(prauc)
            history["med"].append(avg_med)

            if epoch >= 5:
                print(
                    "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                        np.mean(history["ddi_rate"][-5:]),
                        np.mean(history["med"][-5:]),
                        np.mean(history["ja"][-5:]),
                        np.mean(history["avg_f1"][-5:]),
                        np.mean(history["prauc"][-5:]),
                    )
                )

            torch.save(
                model.state_dict(),
                open(os.path.join(self.log, self.model_name + '-' + self.task,
                                  "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                                      epoch, self.target_ddi, ja, ddi_rate), ), "wb", ), )

            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja
                epochs_without_improvement = 0
            elif best_ja >= ja:
                epochs_without_improvement += 1
            print("best_epoch: {}".format(best_epoch))
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. No improvement in {patience} epochs.")
                break
        with open(os.path.join(
                self.log,
                self.model_name + '-' + self.task, "ddi_loss_{}.txt".format(self.model_name)
        ), 'w') as Fout:
            for dloss, dvalue in zip(ddi_losses, ddi_values):
                Fout.write(f'{dloss}\t{dvalue}\n')

        dill.dump(
            history,
            open(
                os.path.join(
                    self.log, self.model_name + '-' + self.task, "history2_{}.pkl".format(self.model_name)
                ),
                "wb",
            ),
        )
