# Measurement-grounded CMR diagnosis under multi-vendor shift

Code for the manuscript *"Why does a cardiac MRI diagnosis model fail at a new hospital?
Measurement-level uncertainty and failure attribution under multi-vendor shift"*.

The easiest way to reproduce everything is the notebook `CMR_diagnosis_full_pipeline.ipynb`:
it follows the paper section by section (Part 1 Data, 3.1 → Part 9 Figures), with a short text before each part
saying which question it answers and which table or section it feeds. The analyses of steps 13–15 appear there
in the sections they belong to (4.3, 4.4, 4.6) and share one helper cell.
The same code is also provided as separate scripts in `scripts/`.

Every script is self-contained, runs in Google Colab or Kaggle, and reads/writes one
working folder (`ACDC_DRIVE`, default `/content/drive/MyDrive/CMR_concept_bottleneck`).
All splits, seeds (42) and inner-validation patients are shared across scripts, so every
model is compared on the same patients.

## Data

| Dataset | Use | Source |
| --- | --- | --- |
| ACDC (100 training patients) | development, 5-fold patient-level CV | https://www.creatis.insa-lyon.fr/Challenge/acdc/ |
| M&Ms-1 (public release, 345 subjects) | locked external test (DCM, HCM, NOR: 271) | https://www.ub.edu/mnms/ |
| CineMA pretrained weights | foundation-model comparator | https://github.com/mathpluscode/CineMA (`pretrained/cinema.safetensors` only) |

Required in the working folder before step 1: `folds.csv` (columns `patient, label, fold`;
5 stratified folds, seed 42). Paths to the ACDC zip and the M&Ms-1 zip are set at the
top of the scripts that need them (`ACDC_ROOT`, glob for `M&M*.zip`).

## Run order

| # | Script | Hardware | Produces (in the working folder) | Paper |
| --- | --- | --- | --- | --- |
| 0 | `step0_inspect_mnms.py` | CPU | layout and metadata of the M&Ms-1 zip (read-only) | 3.1 |
| 1 | `step1_unet_folds.py` | GPU | `step1_unet/` fold-wise U-Nets, out-of-fold Dice | 3.2 |
| 2 | `step3_challenge_baseline.py` | GPU/CPU | `step3_challenge_baseline/` 21 measurements (reference + predicted), predicted masks, random forest | 3.2, 3.4, Table 1 |
| 3 | `step2b_concepts21.py` | CPU | `step2b_concepts21/` main logistic rule | 3.3, Table 1 |
| 4 | `make_heart_centers.py` | CPU | `heart_centers.csv` (from out-of-fold masks) | 3.4 |
| 5 | `step_cinema.py` | GPU | CineMA probe (`MODE=frozen`) and fine-tuning (`MODE=finetune`, `ROI=1`); frozen mode also writes `cinema_features.csv`, `cinema_oof_probs.npy` | 3.4, Table 1 |
| 6 | `step4_joint.py` | GPU | `step4_joint/` joint concept model, concept agreement | 3.4, 4.1 |
| 7 | `step5_c1_additive.py` | CPU | `step5_c1_additive/` additive model | 3.4, Table 1 |
| 8 | `step6_feasibility_laws.py` | CPU | physiological-law check (negative result) | 4.7 |
| 9 | `step7_eiv_conformal.py` | CPU | global Gaussian error model + conformal sets (negative result) | 4.7 |
| 10 | `step8_patient_uncertainty.py` | GPU then CPU | `step8_patient_uncertainty/` 20 stochastic segmentations per patient, kappa, flip flag | 3.5, 3.6, 4.2 |
| 11 | `step8b_bootstrap.py` | CPU | paired bootstrap of step 8 | 4.2 |
| 12 | `step9_mnms_external.py` | GPU then CPU | `step9_mnms_external/mnms_concepts.npz` locked external test | 3.8, 4.3 |
| 13 | `step9b_error_types.py` | CPU | error attribution (segmentation vs rule) | 4.3 |
| 14 | `step9c_phase_fix.py` | CPU | ED/ES rule, final external results, `step9c_per_patient.csv` | 4.3, 4.4, Table 3 |
| 15 | `step10_fewshot_recalibration.py` | CPU | few-shot offsets vs full refit | 3.9, 4.5, Table 4 |
| 16 | `step11_ablation_stats.py` | CPU | ablations, McNemar, Wilcoxon | 4.6, Tables 5-6 |
| 17 | `step12_figures.py` | CPU | Figures 2-5 (PDF + PNG, 300 dpi) | Figs 2-5 |
| 18 | `step13_dcm_strata_threshold.py` | CPU | DCM phenotype strata, flag-threshold curve and nested choice | 4.3, 4.4, 4.6 |
| 19 | `step14_inventory_repeated_cv.py` | CPU | M&Ms-1 label and height/weight inventory, repeated CV of the flag (10 splits) | 4.6 |
| 20 | `step15_bsa_other_classes.py` | CPU | BSA-indexed rule, classes without an ACDC counterpart | 4.3, 4.6 |

GPU steps were run on a single NVIDIA T4 (Kaggle/Colab). Steps that sample (8, 9) cache
their outputs, so a dropped session resumes where it stopped.

## Figure file mapping

| Manuscript | File |
| --- | --- |
| Figure 2 | `step12_figures/fig3_uncertainty_vs_error` |
| Figure 3 | `step12_figures/fig2_flag_roc` |
| Figure 4 | `step12_figures/fig4_error_attribution` |
| Figure 5 | `step12_figures/fig5_fewshot` |

## Leakage rules (enforced in every script)

- Every patient is segmented by the U-Net of the fold in which it was held out.
- Scalers, classifiers, regularisation, kappa and conformal thresholds are fitted on
  training patients only (inner cross-validation where needed).
- The decision rule is fitted on reference measurements of training patients and applied
  to predicted measurements of held-out patients.
- Everything used on M&Ms-1 is frozen on ACDC before the external run. The only change
  made after inspecting M&Ms-1 (ED/ES ordering, step 9c) is reported with both versions.

## Environment

See `requirements.txt`. CineMA is installed from its repository without dependencies to
keep the Colab/Kaggle numpy build:

```
git clone -q https://github.com/mathpluscode/CineMA.git
pip install --no-deps -e CineMA
pip install omegaconf hydra-core monai timm nibabel
```

## Citation

If you use this code, please cite the article (see `CITATION.cff`; GitHub shows a "Cite this repository" button).

## License

MIT, see `LICENSE`. The ACDC and M&Ms-1 datasets and the CineMA weights keep their own licences and are not redistributed here.
