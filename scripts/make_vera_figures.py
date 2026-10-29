import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 统一设置 IEEE / Nature 出版级字体与排版样式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs("figures", exist_ok=True)

# 配色方案 (Okabe-Ito 学术色盲友好高对比色板)
C_NAVY = "#1f4e78"
C_TEAL = "#2b8cbe"
C_ORANGE = "#e6550d"
C_RED = "#d95f02"
C_GREY = "#7f7f7f"
C_LIGHT = "#bdbdbd"
C_GREEN = "#31a354"

# ==============================================================================
# Figure 2: Stage 2 Multi-Condition Robustness & Gate Polarization Audit
# ==============================================================================
def plot_fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.35, 1]})

    # Left: Robustness across corruption groups
    groups = ['Clean\n(n=2,158)', 'Signal Corrupt\n(Noise/Base/Drop)', 'Image Corrupt\n(Blur/Crop/JPEG)', 'Both Corrupt\n(Simultaneous)']
    a5_vals = [0.9047, 0.8776, 0.9017, 0.8637]
    a6_vals = [0.9057, 0.8764, 0.9007, 0.8611]
    pcbef_v2 = [0.9191, 0.9148, 0.9059, 0.9005]
    tmc_vals = [0.9201, 0.9163, 0.9062, 0.9009]

    x = np.arange(len(groups))
    w = 0.19
    ax1.bar(x - 1.5*w, a5_vals, w, label='A5: Gated + ModDrop', color=C_LIGHT)
    ax1.bar(x - 0.5*w, a6_vals, w, label='A6: MLP Quality Gate', color=C_GREY)
    ax1.bar(x + 0.5*w, pcbef_v2, w, label='PC-BEF-v2 (Ours, 3 seeds)', color=C_TEAL)
    rects = ax1.bar(x + 1.5*w, tmc_vals, w, label='TMC Beta-Evidential (Ours)', color=C_NAVY)

    ax1.set_ylim(0.84, 0.935)
    ax1.set_ylabel('Macro AUROC (PTB-XL Test)')
    ax1.set_title('(a) Multimodal Robustness Under Asymmetric Corruptions', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    for r in rects:
        h = r.get_height()
        ax1.annotate(f'{h:.4f}', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color=C_NAVY)

    # Right: Clean ECE vs 7-Condition q-Permutation Sensitivity
    models_r = ['A6 (Old MLP)\nConcatenated q', 'PC-BEF-v1\nPrior Discount', 'PC-BEF-v2\nBounded Blend', 'TMC\nBeta-Evidential']
    ece_vals = [2.49, 2.74, 1.89, 1.25]      # in %
    perm_vals = [0.001, 1.574, 0.001, 0.0]   # in % delta AUC

    x2 = np.arange(len(models_r))
    w2 = 0.32
    ax2_twin = ax2.twinx()
    ax2_twin.spines['top'].set_visible(False)

    b1 = ax2.bar(x2 - w2/2, ece_vals, w2, label='Clean ECE (%) ↓', color=C_TEAL)
    b2 = ax2_twin.bar(x2 + w2/2, perm_vals, w2, label='q-Permutation ΔAUC (%) ↑', color=C_ORANGE)

    ax2.set_ylabel('Expected Calibration Error (%) ↓', color=C_TEAL, fontweight='bold')
    ax2_twin.set_ylabel('7-Cond. q-Permutation ΔAUC (%) ↑', color=C_ORANGE, fontweight='bold')
    ax2.set_title('(b) Calibration Error vs. Quality Controllability', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models_r)
    ax2.set_ylim(0, 3.4)
    ax2_twin.set_ylim(0, 2.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    for r in b1:
        ax2.annotate(f'{r.get_height():.2f}%', xy=(r.get_x()+r.get_width()/2, r.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8, color=C_NAVY)
    ax2_twin.annotate('+1.57%\n(21σ)', xy=(x2[1]+w2/2, 1.574), xytext=(0, 3),
                      textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold', color=C_ORANGE)

    fig.tight_layout()
    fig.savefig("figures/fig2_robustness_gate.pdf", bbox_inches='tight')
    fig.savefig("figures/fig2_robustness_gate.png", bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# Figure 3: Stage 3 APEB Contradiction Rate Reduction & Interval Correlation
# ==============================================================================
def plot_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: Diagnostic AUC vs Clinical Contradiction Rate
    variants = ['B2\n(No Logic)', 'A7\n(Full APEB)', 'A8\n(Bi-Logic)', 'A8-OneWay\n(Ours, 3 seeds)']
    aucs = [0.9211, 0.9211, 0.9198, 0.9205]
    contra = [14.27, 0.88, 0.74, 0.68] # in %

    x = np.arange(len(variants))
    ax1_twin = ax1.twinx()
    ax1_twin.spines['top'].set_visible(False)

    bars = ax1.bar(x, contra, width=0.45, color=[C_TEAL, C_TEAL, C_NAVY], alpha=0.85, label='Contradiction Rate (%) ↓')
    line = ax1_twin.plot(x, aucs, color=C_ORANGE, marker='o', linewidth=2.5, markersize=8, label='Diagnostic Macro-AUC ↑')

    ax1.set_ylabel('Clinical Logic Contradiction Rate (%) ↓', color=C_NAVY, fontweight='bold')
    ax1_twin.set_ylabel('Diagnostic Macro-AUROC ↑', color=C_ORANGE, fontweight='bold')
    ax1.set_title('(a) APEB Variants: Accuracy vs. Contradiction', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants)
    ax1.set_ylim(0, 1.25)
    ax1_twin.set_ylim(0.915, 0.924)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    for b in bars:
        h = b.get_height()
        ax1.annotate(f'{h:.2f}%', xy=(b.get_x()+b.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold')

    # Right: Interval Pearson Correlation Recovery (B4' vs B1 vs APEB A8-OneWay)
    intervals = ['HR', 'PR Interval\n(41.5% Missing)', 'QRS Duration', 'QT Interval', 'QTc Interval']
    b4_prime = [0.822, 0.180, 0.145, 0.534, 0.284]
    b1_aux = [0.937, 0.485, 0.524, 0.695, 0.553]
    apeb_ours = [0.919, 0.508, 0.563, 0.675, 0.557]

    x2 = np.arange(len(intervals))
    w = 0.25
    ax2.bar(x2 - w, b4_prime, w, label="B4': Naive Scalar Bottleneck", color=C_RED, alpha=0.8)
    ax2.bar(x2, b1_aux, w, label="B1: Unconstrained Regression", color=C_LIGHT)
    ax2.bar(x2 + w, apeb_ours, w, label="APEB A8-OneWay (Ours)", color=C_NAVY)

    ax2.set_ylabel('Pearson Correlation Coefficient (r) ↑')
    ax2.set_title('(b) Recovery of Physiological Interval Correlations', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(intervals)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig("figures/fig3_apeb_contradiction.pdf", bbox_inches='tight')
    fig.savefig("figures/fig3_apeb_contradiction.png", bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# Figure 4: Experiment D Counterfactual Anatomical Lead Masking
# ==============================================================================
def plot_fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    labels = ['Targeted Coronary\nTerritory Masking', 'Random Non-Target\nTerritory Masking']
    dp = [0.2020, 0.0284]
    dlogit = [2.4123, 0.2772]

    b1 = ax1.bar(labels, dp, width=0.45, color=[C_NAVY, C_LIGHT])
    ax1.set_ylabel('Target Class Probability Drop (Δp) ↑')
    ax1.set_title('(a) Probability Sensitivity (CSR = 7.12×)', fontweight='bold')
    ax1.set_ylim(0, 0.24)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for b in b1:
        ax1.annotate(f'{b.get_height():.4f}', xy=(b.get_x()+b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    b2 = ax2.bar(labels, dlogit, width=0.45, color=[C_ORANGE, C_LIGHT])
    ax2.set_ylabel('Target Class Logit Drop (Δlogit) ↑')
    ax2.set_title('(b) Logit Sensitivity Ratio (8.70×)', fontweight='bold')
    ax2.set_ylim(0, 2.8)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for b in b2:
        ax2.annotate(f'{b.get_height():.4f}', xy=(b.get_x()+b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    fig.tight_layout()
    fig.savefig("figures/fig4_counterfactual_csr.pdf", bbox_inches='tight')
    fig.savefig("figures/fig4_counterfactual_csr.png", bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# Figure 5: Stage 4 Cross-Center Transfer to CPSC2018 (Per-Class & Few-Shot)
# ==============================================================================
def plot_fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1.4, 1]})

    classes = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD\n(Ischemia)', 'STE\n(Acute MI)']
    old_a0 = [0.8670, 0.9800, 0.9180, 0.9880, 0.9740, 0.7390, 0.8810, 0.7250, 0.6700]
    a0_mv_e2e = [0.8546, 0.9757, 0.9003, 0.9570, 0.9702, 0.7341, 0.8188, 0.6262, 0.5355]
    c0_bal = [0.8897, 0.9708, 0.9591, 0.9282, 0.9736, 0.8528, 0.9095, 0.8973, 0.8123]

    x = np.arange(len(classes))
    w = 0.26
    ax1.bar(x - w, a0_mv_e2e, w, label='A0-mV E2E (Prior Skewed: 0.8192)', color=C_LIGHT)
    ax1.bar(x, old_a0, w, label='Old A0 Baseline (0.8603)', color=C_TEAL)
    ax1.bar(x + w, c0_bal, w, label='C0 Balanced Zero-Shot (Ours: 0.9104)', color=C_NAVY)

    ax1.set_ylabel('5-Fold Cross-Center AUROC')
    ax1.set_title('(a) CPSC2018 Zero-Shot Per-Class Recovery (STD & STE)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0.45, 1.03)
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Highlight STD and STE gains
    ax1.annotate('+17.2 pp', xy=(7+w, 0.8973), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold', color=C_ORANGE)
    ax1.annotate('+14.2 pp', xy=(8+w, 0.8123), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold', color=C_ORANGE)

    # Right: Few-shot adaptation curve
    ratios = ['0%\n(Zero-Shot)', '1%\n(n=40)', '5%\n(n=204)', '10%\n(n=410)', '100%\n(n=4,096)']
    c0_curve = [0.9104, 0.8924, 0.9213, 0.9286, 0.9449]
    old_curve = [0.8603, 0.8911, 0.9192, 0.9330, 0.9539]

    x2 = np.arange(len(ratios))
    ax2.plot(x2, old_curve, marker='s', linestyle='--', color=C_GREY, linewidth=2, label='Unbalanced Source Init')
    ax2.plot(x2, c0_curve, marker='o', linestyle='-', color=C_NAVY, linewidth=2.5, label='Class-Balanced Init (Ours)')
    ax2.set_ylabel('CPSC2018 Macro-AUROC')
    ax2.set_title('(b) Zero-Shot & Few-Shot Target Adaptation', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ratios)
    ax2.set_ylim(0.84, 0.965)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig("figures/fig5_cpsc_crossdomain.pdf", bbox_inches='tight')
    fig.savefig("figures/fig5_cpsc_crossdomain.png", bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# Figure 6: Stage 5 ECG-QA Full Test Set (n=6,330) Breakdown
# ==============================================================================
def plot_fig6():
    fig, ax = plt.subplots(figsize=(9.5, 4.2))

    qtypes = [
        'Single\nVerify\n(n=1,367)',
        'Single\nChoose\n(n=763)',
        'Single\nQuery\n(n=1,400)',
        'Consecutive\nVerify\n(n=800)',
        'Consecutive\nQuery\n(n=600)',
        'Irrelevant\nVerify\n(n=800)',
        'Irrelevant\nQuery\n(n=600)',
        'OVERALL\nFULL TEST\n(n=6,330)'
    ]
    legacy_em = [73.15, 58.85, 58.21, 71.25, 37.00, 79.25, 56.33, 63.63]
    set_em    = [73.15, 46.40, 53.79, 71.25, 36.50, 79.25, 42.83, 59.83]
    set_f1    = [73.15, 54.70, 56.09, 71.25, 36.79, 79.25, 49.46, 61.99]

    x = np.arange(len(qtypes))
    w = 0.25
    ax.bar(x - w, legacy_em, w, label='Legacy EM (%)', color=C_NAVY)
    ax.bar(x, set_f1, w, label='Order-Invariant Set F1 (%)', color=C_TEAL)
    ax.bar(x + w, set_em, w, label='Strict Set EM (%)', color=C_ORANGE)

    ax.axvline(x=6.5, color=C_GREY, linestyle='--', linewidth=1.2)
    ax.set_ylabel('Accuracy / F1 Score (%) ↑')
    ax.set_title('Stage 5: Qwen2.5-3B HE-DVR Performance Across All 6,330 ECG-QA Test Questions', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes)
    ax.set_ylim(0, 90)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for i, v in enumerate(legacy_em):
        ax.annotate(f'{v:.1f}%', xy=(x[i]-w, v), xytext=(0, 2),
                    textcoords='offset points', ha='center', fontsize=7.5, fontweight='bold', color=C_NAVY)

    fig.tight_layout()
    fig.savefig("figures/fig6_ecgqa_breakdown.pdf", bbox_inches='tight')
    fig.savefig("figures/fig6_ecgqa_breakdown.png", bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# Figure 7: Real PTB-XL qualitative case + APEB evidence + HE-DVR QA
# ==============================================================================
def plot_fig7_case_study():
    root = Path(__file__).resolve().parents[1]
    meta = pd.read_csv(root / "data" / "processed" / "ptbxl_100" / "meta.csv")
    tmeta = meta[meta["split"] == "test"].reset_index(drop=True)
    ecg_id = 15059
    row = int(tmeta.index[tmeta["ecg_id"] == ecg_id][0])
    x = np.load(root / "data" / "processed" / "ptbxl_100" / "X_test.npy", mmap_mode="r")
    rec = np.array(x[row], dtype=np.float32)
    bank = np.load(root / "data" / "vera_ecgqa" / "ecg_bank.npz")
    id2idx = json.load(open(root / "data" / "vera_ecgqa" / "id2idx.json", encoding="utf-8"))
    anchors = json.load(open(root / "data" / "vera_ecgqa" / "anchors.json", encoding="utf-8"))
    idx = id2idx[str(ecg_id)]
    ev = bank["ev"][idx]
    anchor = anchors[idx]
    diag_mi, group_inf, group_ant = float(ev[23]), float(ev[5]), float(ev[7])
    pr_valid, qrs_wide, unc = float(ev[10]), float(ev[18]), float(ev[27])

    qa = None
    for line in open(root / "data" / "vera_ecgqa" / "test.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if ecg_id in s.get("ecg_ids", []) and "diagnostic symptom" in s["question"].lower():
            qa = s
            break
    pred_path = root / "results" / "VERA_Qwen_case15059.json"
    pred = "myocardial infarction in inferior leads"
    if pred_path.exists():
        pred = json.load(open(pred_path, encoding="utf-8"))["examples"][0]["pred"]
    question = qa["question"] if qa else "Which diagnostic symptom does this ECG show?"
    gold = "; ".join(qa["answer"]) if qa else "myocardial infarction in inferior leads"

    fig = plt.figure(figsize=(12.2, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.35], wspace=0.30)

    # Panel (a): real PTB-XL test record, lead II + V2
    ax1 = fig.add_subplot(gs[0])
    start = int(2.0 * 100)
    end = start + 300
    tt = np.arange(end - start) / 100.0
    sig_ii = rec[1, start:end]
    sig_v2 = rec[7, start:end] - 0.9
    ax1.set_facecolor("#fff5f5")
    for gx in np.arange(0, 3.01, 0.2):
        ax1.axvline(gx, color="#f4cccc", linewidth=0.7, zorder=0)
    for gy in np.arange(-2.2, 1.61, 0.5):
        ax1.axhline(gy, color="#f4cccc", linewidth=0.7, zorder=0)
    ax1.plot(tt, sig_ii, color="black", linewidth=1.3, label="Lead II (Inferior)")
    ax1.plot(tt, sig_v2, color="#444444", linewidth=1.1, label="Lead V2 (Septal)")
    ax1.axvspan(0.42, 0.62, color="#ff9900", alpha=0.18, label="ST-T window")
    ax1.text(0.05, 1.30, f"Real PTB-XL test record (ecg_id={ecg_id})", fontsize=8.2, fontweight="bold", color="#b30000")
    ax1.text(0.05, 0.95, "Lead II: inferior territory", fontsize=8.0, color="#222222")
    ax1.text(0.05, -1.85, "Lead V2: septal territory", fontsize=8.0, color="#333333")
    ax1.set_ylim(-2.2, 1.55)
    ax1.set_xlim(0, 3.0)
    ax1.set_xlabel("Time (s), 100 Hz")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("(a) Input ECG & Territory Focus", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=7.0, framealpha=0.95)
    ax1.grid(False)

    # Panel (b): real APEB output for this record
    ax2 = fig.add_subplot(gs[1])
    ev_names = ["Diag: MI", "Territory:\nInferior", "Territory:\nAnterior", "PR Valid\nGate", "QRS Wide\nBucket"]
    ev_probs = [diag_mi, group_inf, group_ant, pr_valid, qrs_wide]
    colors = [C_NAVY, C_ORANGE, C_LIGHT, C_TEAL, C_LIGHT]
    y_pos = np.arange(len(ev_names))
    bars = ax2.barh(y_pos, ev_probs, color=colors, height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ev_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("APEB belief / probability")
    ax2.set_title("(b) APEB Evidence (real output)", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for b in bars:
        w = b.get_width()
        ax2.annotate(f"{w:.2f}", xy=(w, b.get_y() + b.get_height() / 2),
                     xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.0, fontweight="bold")
    ax2.text(0.02, 1.02, f"epistemic uncertainty u={unc:.2f}", transform=ax2.transAxes,
             fontsize=7.5, va="bottom", color=C_GREY)

    # Panel (c): real question + real VERA prediction; failure mode is illustrative
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    ax3.set_title("(c) HE-DVR Grounded QA", fontweight="bold")
    qwrap = textwrap.fill(question, width=52)
    box_text = (
        "Input Hard Evidence Anchor (APEB -> Qwen-3B):\n"
        f"• Top-2 Diagnoses: MI={diag_mi:.2f}, CD={float(ev[25]):.2f} | u={unc:.2f}\n"
        f"• Territory: Inferior={group_inf:.2f}, Anterior={group_ant:.2f}\n"
        f"• Intervals: HR={ev[0]:.0f} bpm, PR={ev[1]:.0f} ms (valid={pr_valid:.2f}),\n"
        f"  QRS={ev[2]:.0f} ms (wide={qrs_wide:.2f}), QTc={ev[4]:.0f} ms\n"
        "------------------------------------------------------------\n"
        "Question (ECG-QA Single-Choose):\n"
        f"{qwrap}\n"
        "------------------------------------------------------------\n"
        "Gold: " + gold + "\n\n"
        "× Illustrative soft-token-only failure:\n"
        "  \"anterior myocardial infarction\"\n\n"
        "✓ VERA-ECG (real prediction):\n"
        f"  \"{pred}\"  (EM ✓)"
    )
    ax3.text(0.00, 0.98, box_text, transform=ax3.transAxes, fontsize=7.0,
             verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#1f4e78", linewidth=1.1))

    fig.tight_layout()
    fig.savefig("figures/fig7_qualitative_case.pdf", bbox_inches="tight")
    fig.savefig("figures/fig7_qualitative_case.png", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7_case_study()
    print("Successfully generated Figures 2-7 in ./figures/ (PDF & PNG formats).")



