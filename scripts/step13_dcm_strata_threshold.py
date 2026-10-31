# =============================================================================
# STEP 13 — DCM phenotype strata and flag-threshold justification (CPU, ~3 min, no re-segmentation)
#   A. Stratify the external DCM cohort by phenotype (reference measurements):
#        Classic DCM        : LV EF < 40 %  AND  dilated LV (the ACDC DCM phenotype)
#        Preserved-function : LV EF >= 50 %
#        Intermediate       : everything in between (EF 40-50 %, or EF < 40 % without dilation)
#      "Dilated" = reference LV EDV above the 95th percentile of M&Ms-1 NOR patients
#      (no BSA indexing: height is missing for part of M&Ms-1).
#   B. Justify the flag threshold 0.20 on ACDC:
#        (1) nested choice: in each outer fold, the threshold maximising Youden's J is
#            chosen on the other 80 patients and applied to the held-out 20;
#        (2) full threshold curve (sensitivity, false-flag rate, J) on ACDC and M&Ms-1.
# Same locked model, kappa, ED/ES rule and evaluation as step 9c.
# =============================================================================
import os, io, glob, zipfile, numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
S8, OUT = f'{DRIVE}/step8_patient_uncertainty', f'{DRIVE}/step9_mnms_external'
SEED, S, FLAG = 42, 20, 0.20
PRIMARY = ['DCM', 'HCM', 'NOR']
pd.set_option('display.width', 240)

# ---------------- locked ACDC model (identical to step 9c)
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y_acdc = np.searchsorted(classes, folds.label.values)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
F = {c: j for j, c in enumerate(FEATS)}
A_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
A_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
A_cs = np.load(f'{S8}/concept_samples.npz')['CS']
head = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                    {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy').fit(A_gt, y_acdc).best_estimator_
KAPPA = float(np.median(np.abs(A_pr - A_gt) / (A_cs.std(1) + 1e-9)) / 0.6745)
pidx = np.searchsorted(classes, PRIMARY)

# ---------------- M&Ms measurements + ED/ES rule (identical to step 9c)
R = pd.DataFrame(list(np.load(f'{OUT}/mnms_concepts.npz', allow_pickle=True)['res']))
z = zipfile.ZipFile(glob.glob('/content/drive/MyDrive/**/M&M*.zip', recursive=True)[0])
meta = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if 'MnMs_Dataset_information' in n][0])))
meta['External code'] = meta['External code'].astype(str).str.strip()
R['centre'] = R.code.map(meta.set_index('External code')['Centre'])
n = len(R)
Cg, Cp, CS = np.stack(list(R['gt'])), np.stack(list(R['pred'])), np.stack(list(R['samples']))

def swap_phases(X):
    X = X.copy()
    for a in ('RV', 'MYO', 'LV'):
        X[..., [F[f'{a}_ED'], F[f'{a}_ES']]] = X[..., [F[f'{a}_ES'], F[f'{a}_ED']]]
    for s in ('max', 'mean', 'std', 'min'):
        X[..., [F[f'WT_ED_{s}'], F[f'WT_ES_{s}']]] = X[..., [F[f'WT_ES_{s}'], F[f'WT_ED_{s}']]]
    e = 1e-6
    g = lambda c: X[..., F[c]]
    X[..., F['LV_EF']] = 100 * (g('LV_ED') - g('LV_ES')) / np.maximum(g('LV_ED'), e)
    X[..., F['RV_EF']] = 100 * (g('RV_ED') - g('RV_ES')) / np.maximum(g('RV_ED'), e)
    X[..., F['LV_mass']] = 1.05 * g('MYO_ED')
    X[..., F['RV_LV_ED']] = g('RV_ED') / np.maximum(g('LV_ED'), e); X[..., F['RV_LV_ES']] = g('RV_ES') / np.maximum(g('LV_ES'), e)
    X[..., F['MYO_LV_ED']] = g('MYO_ED') / np.maximum(g('LV_ED'), e); X[..., F['MYO_LV_ES']] = g('MYO_ES') / np.maximum(g('LV_ES'), e)
    return X

sw = Cp[:, F['LV_ED']] < Cp[:, F['LV_ES']]
Cg[sw], Cp[sw], CS[sw] = swap_phases(Cg[sw]), swap_phases(Cp[sw]), swap_phases(CS[sw])

def p3(X):
    L = head.decision_function(X)[:, pidx]
    return np.exp(L - logsumexp(L, 1, keepdims=True))
yv = np.searchsorted(PRIMARY, R.label.values); cen = R.centre.values
dv = CS - CS.mean(1, keepdims=True)
pred, ref = p3(Cp).argmax(1), p3(Cg).argmax(1)
wrong, ref_ok = pred != yv, ref == yv
typ = np.where(~wrong, 'CORRECT', np.where(ref_ok, 'SEG', 'RULE'))
flips = np.zeros((n, len(FEATS)))
for t in range(n):
    for j in range(len(FEATS)):
        X = np.repeat(Cp[[t]], S, 0); X[:, j] += KAPPA * dv[t, :, j]
        flips[t, j] = np.mean(p3(X).argmax(1) != pred[t])
fmax_m = flips.max(1)

# =============================================================================
# A. DCM phenotype strata
# =============================================================================
EF, EDV = Cg[:, F['LV_EF']], Cg[:, F['LV_ED']]
nor = R.label.values == 'NOR'
EDV_ULN = np.percentile(EDV[nor], 95)
dcm = R.label.values == 'DCM'
stratum = np.full(n, '', dtype=object)
stratum[dcm & (EF < 40) & (EDV > EDV_ULN)] = 'Classic DCM'
stratum[dcm & (EF >= 50)] = 'Preserved-function DCM'
stratum[dcm & (stratum == '')] = 'Intermediate DCM'

acdc_dcm = folds.label.values == 'DCM'
print(f'LV EDV upper limit (95th pct of M&Ms-1 NOR, reference): {EDV_ULN:.0f} mL')
print(f'ACDC DCM (reference): median LV EF {np.median(A_gt[acdc_dcm, F["LV_EF"]]):.1f} %, '
      f'median LV EDV {np.median(A_gt[acdc_dcm, F["LV_ED"]]):.0f} mL, '
      f'{np.mean((A_gt[acdc_dcm, F["LV_EF"]] < 40) & (A_gt[acdc_dcm, F["LV_ED"]] > EDV_ULN)) * 100:.0f} % meet the classic definition')

rows = []
for s in ['Classic DCM', 'Intermediate DCM', 'Preserved-function DCM']:
    m = stratum == s
    if m.sum() == 0:
        continue
    rows.append(dict(stratum=s, n=m.sum(), median_EF=np.median(EF[m]), median_EDV=np.median(EDV[m]),
                     recall_pred=np.mean(pred[m] == 0), recall_refmeas=np.mean(ref[m] == 0),
                     errors=wrong[m].sum(), SEG=(typ[m] == 'SEG').sum(), RULE=(typ[m] == 'RULE').sum(),
                     called_NOR=np.sum(pred[m] == 2), called_HCM=np.sum(pred[m] == 1)))
T = pd.DataFrame(rows)
print('\nA. External DCM patients by phenotype (recall = share correctly called DCM):')
print(T.round(2).to_string(index=False))
print('\nDCM strata by centre:')
print(pd.crosstab(stratum[dcm], cen[dcm], rownames=['stratum'], colnames=['centre']).to_string())
print('\nCentre 3 DCM vs M&Ms-1 NOR (reference): median LV EDV '
      f'{np.median(EDV[dcm & (cen == 3)]):.0f} vs {np.median(EDV[nor]):.0f} mL; '
      f'dilated (> ULN): {np.mean(EDV[dcm & (cen == 3)] > EDV_ULN) * 100:.0f} % of centre-3 DCM')

# =============================================================================
# B. Flag threshold
# =============================================================================
ex = pd.read_csv(f'{S8}/doubt_per_patient.csv').set_index('patient').loc[order]
fmax_a = ex.doubt1_flip.values
err_a = (ex.label != ex.pred).values
GRID = np.round(np.arange(0.05, 0.501, 0.05), 2)          # multiples of 1/S

def curve(f, e):
    return pd.DataFrame([dict(threshold=t, sensitivity=np.mean(f[e] >= t), false_flag=np.mean(f[~e] >= t),
                              youden=np.mean(f[e] >= t) - np.mean(f[~e] >= t)) for t in GRID])

print('\nB1. Threshold curve, ACDC (9 errors, 91 correct):')
print(curve(fmax_a, err_a).round(2).to_string(index=False))
seg_or_ok = (typ == 'SEG') | (typ == 'CORRECT')
print('\nB2. Threshold curve, M&Ms-1, segmentation-driven errors vs correct:')
print(curve(fmax_m[seg_or_ok], (typ == 'SEG')[seg_or_ok]).round(2).to_string(index=False))

print('\nB3. Nested choice on ACDC (threshold chosen on the 80 training patients of each fold):')
chosen, hit = [], np.zeros(len(order), bool)
for k in sorted(folds.fold.unique()):
    te = folds.fold.values == k
    c = curve(fmax_a[~te], err_a[~te])
    best = c.threshold[c.youden == c.youden.max()].min()   # ties -> smaller threshold
    chosen.append(best); hit[te] = fmax_a[te] >= best
print('  thresholds chosen per fold:', chosen)
print(f'  out-of-fold sensitivity {np.mean(hit[err_a]):.2f} ({hit[err_a].sum()}/{err_a.sum()}), '
      f'false-flag rate {np.mean(hit[~err_a]):.2f} ({hit[~err_a].sum()}/{(~err_a).sum()})')
print(f'  fixed 0.20 for comparison: sensitivity {np.mean(fmax_a[err_a] >= FLAG):.2f}, '
      f'false-flag rate {np.mean(fmax_a[~err_a] >= FLAG):.2f}')

T.to_csv(f'{OUT}/step13_dcm_strata.csv', index=False)
print('\nSaved to', OUT)
