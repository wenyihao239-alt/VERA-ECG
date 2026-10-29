# CBM Editorial Manager copy-paste

## Title
VERA-ECG: Verifiable Evidential Reasoning and Anatomy-Grounded Multimodal Framework for Reliable 12-Lead ECG Interpretation

## Authors
Yihao Wen (corresponding), Jun Zeng

## ORCID
Yihao Wen: 0009-0004-7020-3871
Jun Zeng: 0009-0001-4584-945X

## Abstract
Multimodal large language models (MLLMs) combining ECG time series, waveform images, and text are promising but face modality polarization, non-random interval missingness, loss of millivolt scale, and unreliable clinical grounding. We propose VERA-ECG, a verifiable evidential and anatomy-grounded framework. Multi-label Beta-evidential fusion (TMC/PC-BEF) models class-wise positive/negative evidence and epistemic uncertainty with hardware-quality discounting. An anatomy-grouped presence-gated evidence bottleneck (APEB) converts missing intervals into presence-gated features and enforces a one-way differentiable clinical logic loss (A8-OneWay) across four coronary territories. Class-balanced feature calibration addresses cross-center prior shift. Soft multimodal tokens, including longitudinal differential tokens, are combined with hard symbolic anchors in a 3B MLLM (HE-DVR). On PTB-XL, CPSC2018, and ECG-QA, VERA-ECG achieves 0.9205 ± 0.0006 macro-AUROC and 0.68% diagnostic-evidence contradiction (A7: 0.88%), with a counterfactual sensitivity ratio of 7.12 (Δlogit ratio 8.70). C0 reaches 0.9104 five-fold mean CPSC2018 macro-AUROC, improving same-fold Fold0 from 0.8603 to 0.9080 and ST-elevation AUROC from 0.670 to 0.8123. HE-DVR reaches 63.63% legacy and 59.83% set exact match on 6,330 ECG-QA questions.

## Keywords
Electrocardiogram (ECG), Multimodal Large Language Models, Evidential Deep Learning, Neuro-Symbolic Reasoning, Cross-Domain Generalization.

## Highlights
- A Beta-evidential framework models epistemic uncertainty in multi-label ECG.
- APEB resolves non-random interval missingness via clinical presence gating.
- One-way logic loss cuts clinical contradictions from 0.88% to 0.68%.
- Class-balanced probing rescues cross-center transfer without mV bias.
- HE-DVR injects hard symbolic anchors into a 3B MLLM for grounded QA.

## Code / data
https://github.com/wenyihao239-alt/VERA-ECG

## Corresponding author
Yihao Wen
Taiyuan University of Technology
79 Yingze West Street, Taiyuan 030024, Shanxi, China
wenyihao239@gmail.com
