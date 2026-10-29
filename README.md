# VERA-ECG

Code and evaluation protocols for:

**VERA-ECG: Verifiable Evidential Reasoning and Anatomy-Grounded Multimodal Framework for Reliable 12-Lead ECG Interpretation**

Yihao Wen, Jun Zeng

## What is included

- `scripts/`: core training/evaluation code for A0-mV, A1-Pre, TMC/PC-BEF, APEB/A8-OneWay, CPSC calibration, and Qwen ECG-QA.
- `splits/`: PTB-XL split mapping, CPSC fold-level counts, and ECG-QA patient-overlap audit.
- `results/`: reported metrics and full-test ablation JSON files.
- `figures/`: Figures 2-7 used in the manuscript.
- `paper/`: Elsevier/CBM submission files (manuscript, title page, highlights, cover letter, supplementary material).

Raw datasets, rendered images, model checkpoints, and LoRA adapters are not redistributed here. They are public or regenerable using the commands below.

## Data

- **PTB-XL**: https://physionet.org/content/ptb-xl/1.0.3/
- **CPSC2018**: public challenge data; see the challenge distribution.
- **ECG-QA**: https://github.com/JingerLove/ecg-qa

Expected local layout:

```text
data/
  raw/ptbxl/
  processed/ptbxl_100/
  rendered/ptbxl_320/
  cpsc2018_100/
  cpsc2018_fold1_100/ ... cpsc2018_fold4_100/
  vera_features/
  vera_apeb/
  evidence/
  vera_ecgqa/
features/
  A0_cpsc_9/
mydownload/                 # Qwen2.5-3B-Instruct
runs_vera/                  # training outputs
```

## Environment

Python 3.14, PyTorch 2.14.0+cu130, CUDA 13.0, RTX 4060 Ti 16 GB. Install:

```bash
pip install -r requirements.txt
```

## Main commands

```bash
# Stage 1: A0-mV / A1-Pre
python scripts/train_signal_mv.py --data-dir data/processed/ptbxl_100 --mv-dir data/processed/ptbxl_100_mv --task super --model resnet1d --epochs 100 --batch-size 128 --lr 1e-3 --seed 42 --out-dir runs_vera/A0_mV_seed42
python scripts/train_image_pre.py --data-dir data/processed/ptbxl_100 --render-dir data/rendered/ptbxl_320 --task super --model resnet34 --height 384 --width 512 --epochs 30 --batch-size 32 --lr 1e-4 --seed 42 --out-dir runs_vera/A1_Pre_seed42

# Stage 2: TMC / PC-BEF
python scripts/vera_train_pcbef.py --feature-root data/vera_features --data-dir data/processed/ptbxl_100 --mode tmc --epochs 100 --batch-size 512 --lr 1e-3 --seed 42 --out-dir runs_vera/TMC_seed42

# Stage 3: APEB / A8-OneWay
python scripts/vera_train_apeb.py --feature-root data/vera_features --data-dir data/processed/ptbxl_100 --apeb-dir data/vera_apeb --evidence-dir data/evidence --mode a8_oneway --epochs 100 --batch-size 512 --lr 1e-3 --lambda-logic 0.1 --seed 42 --out-dir runs_vera/APEB_a8_oneway_seed42

# Stage 4: CPSC calibration
python scripts/vera_eval_cpsc_mv.py --checkpoint runs_vera/A0_mV_cpsc_seed42/best.pt --root data --out results/VERA_CPSC_A0mV_5fold.json --mv-mode raw
python scripts/vera_coral_calib.py --feature-dir features/A0_cpsc_9 --out results/VERA_CPSC_C0_C2_C3_A0.json

# Stage 5: Qwen SFT + full evaluation
python scripts/vera_train_qwen.py --model-path mydownload --bank data/vera_ecgqa/ecg_bank.npz --qa-dir data/vera_ecgqa --out-dir runs_vera/Qwen_SFT --epochs 1 --max-len 768 --grad-accum 8 --lr 2e-4
python scripts/vera_eval_qwen_full.py --model-path mydownload --adapter runs_vera/Qwen_SFT/adapter --projector runs_vera/Qwen_SFT/projector.pt --bank data/vera_ecgqa/ecg_bank.npz --qa-dir data/vera_ecgqa --answers-csv third_party/ecg-qa-master/ecgqa/ptbxl/answers_for_each_template.csv --out results/VERA_Qwen_final_fulltest.json --all --batch-size 8 --max-new-tokens 64
```

## Full-data LLM ablations

`results/VERA_Qwen_full_noanchor.json`, `VERA_Qwen_full_noimage.json`, and `VERA_Qwen_full_signal.json` are full 1-epoch retrained variants. Their commands are in `paper/` and `scripts/run_reviewer_llm_ablations.ps1`.

## Split and leakage audit

- PTB-XL official patient-isolated train/val/test mapping: `splits/ptbxl_ecg_split.json`.
- ECG-QA train/test patient overlap: `splits/ecgqa_patient_overlap.json` (0 overlap; 9,100 vs. 2,444 patients).
- CPSC fold counts: `splits/cpsc_folds.json` (record-level folds; public data have no patient IDs).

## Reproducibility notes

- Metrics use unweighted macro averages; F1 threshold = 0.5; ECE uses 15 bins.
- C0 is source-only; C2/C3 use unlabeled target-fold features (transductive).
- Temperature scaling is fitted on PTB-XL validation and frozen.
- CPSC 500→100 Hz uses polyphase anti-aliasing (`resample_poly`, 1/5); no extra µV→mV scaling is applied.
- The full Qwen ablations use one epoch (3,584 steps) on 28,672 training pairs.

## License

MIT. Dataset licenses remain with the original providers.

## Contact

Yihao Wen, Taiyuan University of Technology, wenyihao239@gmail.com  
Jun Zeng, Chongqing University of Posts and Telecommunications, zeng.cqupt@gmail.com
