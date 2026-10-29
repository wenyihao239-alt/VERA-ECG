$ErrorActionPreference = "Continue"
$base = "C:\Users\YOLO\Documents\心电图\QEG-ECG"
$runs = @(
    @{name="noanchor"; train="--drop-anchor"; eval="--drop-anchor"},
    @{name="noimage";  train="--zero-image";  eval="--drop-image"},
    @{name="signal";   train="--signal-only"; eval="--signal-only"}
)
foreach ($r in $runs) {
    $out = Join-Path $base "runs_vera\Qwen_full_$($r.name)"
    $log = Join-Path $base "logs\Qwen_full_$($r.name).out.log"
    $err = Join-Path $base "logs\Qwen_full_$($r.name).err.log"
    Write-Output "=== START $($r.name) $(Get-Date -Format o) ==="
    python (Join-Path $base "scripts\vera_train_qwen.py") `
        --model-path (Join-Path $base "mydownload") `
        --bank (Join-Path $base "data\vera_ecgqa\ecg_bank.npz") `
        --qa-dir (Join-Path $base "data\vera_ecgqa") `
        --out-dir $out --epochs 1 --max-steps 3584 --max-len 768 `
        --grad-accum 8 --lr 2e-4 --n-soft 8 --save-steps 1000 --device cuda:0 $r.train 2> $err | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Output "TRAIN FAILED $($r.name)"; continue }
    $evout = Join-Path $base "results\VERA_Qwen_full_$($r.name).json"
    Write-Output "=== EVAL $($r.name) $(Get-Date -Format o) ==="
    python (Join-Path $base "scripts\vera_eval_qwen_full.py") `
        --model-path (Join-Path $base "mydownload") `
        --adapter (Join-Path $out "adapter") `
        --projector (Join-Path $out "projector.pt") `
        --bank (Join-Path $base "data\vera_ecgqa\ecg_bank.npz") `
        --qa-dir (Join-Path $base "data\vera_ecgqa") `
        --answers-csv (Join-Path $base "third_party\ecg-qa-master\ecgqa\ptbxl\answers_for_each_template.csv") `
        --out $evout --all --batch-size 8 --max-new-tokens 64 --device cuda:0 $r.eval 2> $err
    Write-Output "=== DONE $($r.name) $(Get-Date -Format o) ==="
}
Write-Output "ALL DONE $(Get-Date -Format o)"
