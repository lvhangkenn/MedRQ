# MedRQ
This is the official implementation of **Towards Efficient and Interpretable Medical Concept Representation via Ontology-driven Residual Vector Quantization**.


## Data
In compliance with PhysioNet Clinical Database usage requirements, we are unable to share the dataset directly. Researchers interested in accessing the data can request permission via the following resources:  [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/).

The structure of the data set should be like,
```text
dataset
|_ raw
|  |_ DIAGNOSES_ICD.csv
|  |_ PRESCRIPTIONS.csv
|  |_ PROCEDURES_ICD.csv
|  |_ drug-atc.csv
|  |_ drug-DDI.csv
|  |_ ndc2atc_level4.csv
|  |_ ndc2RXCUI.txt
|  |_ precudure_icd9_ontology.csv
|  |_ drug_atc_ontology.csv
|  |_ diagnosis_icd9_ontology.csv
|  |_ D_ICD_PROCEDURES.csv
|  |_ D_ICD_DIAGNOSES.csv
```

After processing by using the scripts provided in this repository,  the structure of the data set should be like(take mimic-iii as an example)


```text
dataset/mimic-iii_data/
  ddi_A_final.pkl          
  records_final.pkl        
  voc_final.pkl            
  ontology/              
    diag/
      icd_vocab_order.txt
      icd_items.jsonl
      tag_vocab_l2.txt
      tag_vocab_l3.txt
      tag_vocab_l4.txt
      tag_vocab_l5.txt
      tags_indices.pt
      tag_class_counts.json
      embeddings/
        icd_text_emb.pt
        tags_emb_l2.pt
        tags_emb_l3.pt
        tags_emb_l4.pt
        tags_emb_l5.pt
      hidvae_sid.pt
      hidvae_sum_emb.pt
      processed/
        medical.pt
    drug/
      icd_vocab_order.txt        
      drug_items.jsonl
      tag_vocab_l2.txt
      tag_vocab_l3.txt
      tag_vocab_l4.txt
      tag_vocab_l5.txt
      tags_indices.pt
      tag_class_counts.json
      embeddings/
        icd_text_emb.pt           
        tags_emb_l2.pt
        tags_emb_l3.pt
        tags_emb_l4.pt
        tags_emb_l5.pt
      hidvae_sid.pt
      hidvae_sum_emb.pt
      processed/
        medical.pt
    proc/
      icd_vocab_order.txt      
      proc_items.jsonl
      tag_vocab_l2.txt
      tag_vocab_l3.txt
      tag_vocab_l4.txt
      tag_vocab_l5.txt
      tags_indices.pt
      tag_class_counts.json
      embeddings/
        icd_text_emb.pt
        tags_emb_l2.pt
        tags_emb_l3.pt
        tags_emb_l4.pt
        tags_emb_l5.pt
      hidvae_sid.pt
      hidvae_sum_emb.pt
      processed/
        medical.pt
```

## Running the code
To run the code, execute the following command:

### Data preprocessing
```bash
cd ../dataset/scripts/process_from_raw_data
python process.py 
```
```bash
cd ../dataset/scripts/prepare_mimic3_ontology_for_medrq
python mimic3_diag_omtology_for_medrq.py \
  --voc-path ../mimic-iii_data/voc_final.pkl \
  --ontology-csv ../raw/procedure_icd9_ontology.csv \
  --out-dir ../mimic-iii_data/ontology/diag\
  --embed\
  --device cuda 

```

### Training 
#### Stage 1: Ontology-driven RQ-VAE
1. training
```bash
cd ../MedRQ_VAE
python train_medrq.py configs/medrq_mimic3_diag.gin
```

2. export semantic id 
```bash
cd ../dataset/scripts
python export_medrq_sid.py \
```

#### Stage 2: Downstream Task-Medication Recommendation
```bash
cd ../downstream_GRU4CMR
python main.py \
```










