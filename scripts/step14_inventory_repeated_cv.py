# =============================================================================
# STEP 14 — data inventory and repeated cross-validation of the flag (CPU, ~5 min, no re-segmentation)
#   Part 0. Inventory (what the extra analyses can use):
#           M&Ms-1 labels outside DCM/HCM/NOR, and height/weight availability
#           (for BSA) in M&Ms-1 and ACDC.
#   Part 1. repeated cross-validation of the flip flag on ACDC.
#           The decision rule is re-fitted on 10 different 5-fold splits (seeds 0-9);
#           the stochastic measurement samples of step 8 are reused unchanged
#           (each mask was produced by a U-Net that never saw that patient).
# =============================================================================
import os, io, re, glob, zipfile, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
S8 = f'{DRIVE}/step8_patient_uncertainty'
OUT = f'{DRIVE}/step14_robustness'; os.makedirs(OUT, exist_ok=True)
S, FLAG, SEEDS = 20, 0.20, range(10)
pd.set_option('display.width', 240)

# =============================================================================
# Part 0. Inventory
# =============================================================================
zm = zipfile.ZipFile(glob.glob('/content/drive/MyDrive/**/M&M*.zip', recursive=True)[0])
meta = pd.read_csv(io.BytesIO(zm.read([n for n in zm.namelist() if 'MnMs_Dataset_information' in n][0])))
print('M&Ms-1 metadata columns:', list(meta.columns))
gt_codes = {re.search(r'([A-Z0-9]{6})_sa_gt\.nii', n).group(1) for n in zm.namelist() if re.search(r'[A-Z0-9]{6}_sa_gt\.nii', n)}
meta['External code'] = meta['External code'].astype(str).str.strip()
m_gt = meta[meta['External code'].isin(gt_codes)]
print(f'\nM&Ms-1 subjects with contours: {len(m_gt)}')
print('Pathology x centre (subjects with contours):')
print(pd.crosstab(m_gt['Pathology'], m_gt['Centre'], margins=True).to_string())
hcol = [c for c in meta.columns if 'height' in c.lower()]
wcol = [c for c in meta.columns if 'weight' in c.lower()]
if hcol and wcol:
    has = m_gt[hcol[0]].notna() & m_gt[wcol[0]].notna() & (pd.to_numeric(m_gt[hcol[0]], errors='coerce') > 0)
    print(f'\nHeight and weight available: {has.sum()}/{len(m_gt)} subjects')
    print(pd.crosstab(m_gt['Pathology'], has, colnames=['has height+weight']).to_string())
    print(pd.crosstab(m_gt['Centre'], has, colnames=['has height+weight']).to_string())
else:
    print('\nNo height/weight columns found in the M&Ms-1 metadata.')

az = [p for p in glob.glob('/content/drive/MyDrive/**/*.zip', recursive=True)
      if any(n.endswith('patient001/Info.cfg') for n in zipfile.ZipFile(p).namelist())]
if az:
    za = zipfile.ZipFile(az[0])
    rows = []
    for nme in za.namelist():
        mm = re.search(r'(patient(\d{3}))/Info\.cfg$', nme)
        if mm and int(mm.group(2)) <= 100:                    # training patients 001-100
            d = dict(l.split(':', 1) for l in za.read(nme).decode().splitlines() if ':' in l)
            rows.append(dict(patient=mm.group(1), height=float(d.get('Height', 'nan')), weight=float(d.get('Weight', 'nan'))))
    acdc_hw = pd.DataFrame(rows).drop_duplicates('patient')
    ok = acdc_hw.height.gt(0) & acdc_hw.weight.gt(0)
    print(f'\nACDC training patients with height and weight: {ok.sum()}/{len(acdc_hw)}  (zip: {az[0]})')
    acdc_hw.to_csv(f'{OUT}/acdc_height_weight.csv', index=False)
else:
    print('\nACDC zip with Info.cfg not found in Drive.')

# =============================================================================
# Part 1. Repeated cross-validation of the flip flag (ACDC)
# =============================================================================
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
C_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
C_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
CS = np.load(f'{S8}/concept_samples.npz')['CS']
dev = CS - CS.mean(1, keepdims=True)
sd = CS.std(1) + 1e-9
n, D = C_pr.shape

def fit_head(idx, seed):
    g = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                     {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                     cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring='balanced_accuracy')
    return g.fit(C_gt[idx], y[idx]).best_estimator_

rows, pooled = [], []
for seed in SEEDS:
    pred, fmax, conf = np.zeros(n, int), np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(C_pr, y):
        mdl = fit_head(tr, seed)
        kap = float(np.median(np.abs(C_pr[tr] - C_gt[tr]) / sd[tr]) / 0.6745)
        P = mdl.predict_proba(C_pr[te]); pred[te] = P.argmax(1); conf[te] = 1 - P.max(1)
        for i, t in enumerate(te):
            fl = []
            for j in range(D):
                X = np.repeat(C_pr[[t]], S, 0); X[:, j] += kap * dev[t, :, j]
                fl.append(np.mean(mdl.predict(X) != pred[t]))
            fmax[t] = max(fl)
    err = pred != y
    rows.append(dict(seed=seed, bal_acc=balanced_accuracy_score(y, pred), errors=err.sum(),
                     AUROC_flag=roc_auc_score(err, fmax), AUROC_conf=roc_auc_score(err, conf),
                     sens_020=np.mean(fmax[err] >= FLAG), false_flag_020=np.mean(fmax[~err] >= FLAG)))
    pooled += [dict(seed=seed, err=e, fmax=f_, conf=c_) for e, f_, c_ in zip(err, fmax, conf)]
    print(f'seed {seed}: {rows[-1]}')

T = pd.DataFrame(rows)
print('\nPart 1. Repeated 5-fold CV (10 seeds), ACDC:')
print(T.round(3).to_string(index=False))
print('\nMean (SD) over seeds:')
print(pd.DataFrame({'mean': T.drop(columns='seed').mean(), 'sd': T.drop(columns='seed').std()}).round(3).to_string())
Pd = pd.DataFrame(pooled)
print(f'\nPooled over seeds: {Pd.err.sum()} error instances in {len(Pd)} decisions; '
      f'flag sensitivity {np.mean(Pd.fmax[Pd.err] >= FLAG):.2f}, false-flag {np.mean(Pd.fmax[~Pd.err] >= FLAG):.2f}; '
      f'flag better than confidence in {np.mean(T.AUROC_flag > T.AUROC_conf) * 100:.0f} % of seeds')
T.to_csv(f'{OUT}/step14_repeated_cv.csv', index=False)
print('\nSaved to', OUT)
