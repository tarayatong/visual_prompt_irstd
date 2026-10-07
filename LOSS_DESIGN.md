# Weighted BCE + Dice + FP/FN hard mining

Training uses a 10-epoch curriculum: epochs 1–3 optimize `BCE + Dice`; epochs
4–10 switch to the full objective below, adding the weighted hard FP/FN term.
The validation loss follows the same epoch schedule, so its numeric value may
jump at epoch 4 even if segmentation quality changes smoothly. Compare IoU,
Pd, and Fa across that boundary for quality trends.

The active objective is `balanced_hard`. It uses binary cross entropy, ordinary
symmetric Soft Dice, and a symmetric hard-error term. Tversky and GCE mixing are
not used.

For one image, P={i:y_i=1}, N={i:y_i=0}, n_pos=|P|, n_neg=|N|:

    L_pos = sum_{i in P} -log(p_i) / n_pos
    L_bg  = sum_{i in N} w_i * [-log(1-p_i)] / n_neg
    K_FP  = min(n_neg, max(64, floor(5*n_pos)))
    H_FP  = indices of the K_FP largest p_i among N
    K_FN  = min(n_pos, max(1, ceil(0.25*n_pos)))
    H_FN  = indices of the K_FN smallest p_i among P
    L_FP  = sum_{i in H_FP} w_i * [-log(1-p_i)] / K_FP
    L_FN  = sum_{i in H_FN} -log(p_i) / K_FN
    L_hard = 0.5 * (L_FP + L_FN)
    L_dice = 1 - (2*sum(p*y)+1)/(sum(p)+sum(y)+1)
    L_total = L_pos + 0.25*L_bg + 0.5*L_hard + 1.0*L_dice

All terms are calculated per image and then averaged over the batch. FP and FN
hard terms are averaged separately before the 50/50 combination, so abundant
background errors cannot drown out rare missed-target errors. Top-K selection is
detached; selected BCE values remain differentiable. A hard FP is a labeled
background pixel with high target probability. A hard FN is a labeled foreground
pixel with low target probability. Empty/full masks safely omit a hard term for
its absent class; other terms remain valid.

For an optional exterior one-pixel ring R=dilate_3x3(P) minus P, w_i=rho inside
R and 1 elsewhere. Default rho=1 disables this adjustment; rho=0.5 enables it.
Denominators are pixel counts, not weight sums. The FP ranking uses raw
probabilities. No foreground label is eroded.

Run the independent 10-epoch comparison with:

```powershell
python -m scripts.train_lora --config experiments/configs/train_lora_r8_balanced_hard.yaml
```

Output: `outputs/training/mobilesam_kv_r8_balanced_hard`. The sparse_robust YAML
filename is retained as a compatibility path to the same FP/FN hard objective;
the removed sparse_robust loss itself is unsupported. Historical checkpoints
that use the removed objective fail explicitly rather than silently changing
loss. Resume rejects changed settings.

Logs report total loss and component diagnostics (`bce`, `dice`, `foreground`,
`background`, `hard_fp`, `hard_fn`, `hard`). Diagnostic components are unweighted;
training total uses coefficients above. Absolute total loss is not comparable
between different objectives.

This is a task-specific proposed objective; gains are unverified. The mask remains
a hard target for BCE, so mislabeled pixels can still be strongly penalized. The
optional boundary ring only addresses a narrow uncertainty assumption. Validation
metrics use the original masks. These pixel-level losses are not direct Pd/Fa
objectives.

Run `python -m scripts.check_sparse_loss` for numerical formula/gradient checks.
