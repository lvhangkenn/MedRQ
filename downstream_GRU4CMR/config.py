
class MedAlignConfig():
    MODEL = "GRU"
    TASK = 'MIII'
    RATIO = 2/3

    SEED = 2023
    USE_CUDA = True
    GPU_ONLY = True
    GPU = '0'
    EPOCH = 1000
    DIM = 64
    LR = 5e-4 
    BATCH = 32
    WD = 0
    DDI = 0.06
    KP = 0.08
    HIST = 3 

    ROOT = '../MedRQ/dataset/'
    LOG = '../log/'
    ontology_ROOT = '../MedRQ/dataset/mimic-iii_data/ontology/'
    # semantic embedding 配置
    HIDVAE_DIAG_EMB = '../MedRQ/dataset/mimic-iii_data/ontology/diag/hidvae_sum_emb.pt'
    HIDVAE_PROC_EMB = '../MedRQ/dataset/mimic-iii_data/ontology/proc/hidvae_sum_emb.pt'
    HIDVAE_DRUG_EMB = '../MedRQ/dataset/mimic-iii_data/ontology/drug/hidvae_sum_emb.pt'
    # semantic id 配置
    HIDVAE_USE_SID = True
    HIDVAE_DIAG_SID = '../MedRQ/dataset/mimic-iii_data/ontology/diag/hidvae_sid.pt'
    HIDVAE_PROC_SID = '../MedRQ/dataset/mimic-iii_data/ontology/proc/hidvae_sid.pt'
    HIDVAE_DRUG_SID = '../MedRQ/dataset/mimic-iii_data/ontology/drug/hidvae_sid.pt'

    SID_AGGREGATION = 'concat'

    #测试
    PERF_TEST = False
    PERF_TEST_THRESHOLD = 0.5
    PERF_TEST_WARMUP_ROUNDS = 1

    #few-shot模式
    FEWSHOT = False
    FEWSHOT_RATIO = 0.05 



config = vars(MedAlignConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}
