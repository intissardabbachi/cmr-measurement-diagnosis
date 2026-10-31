# =============================================================================
# STEP 15 — BSA indexing and classes without an ACDC counterpart (CPU, ~5-10 min, no re-segmentation)
#   Part A. volumes and mass indexed to body surface area (Mosteller).
#           Rule refitted on indexed ACDC reference measurements and compared with the
#           locked (non-indexed) rule on the M&Ms-1 primary patients that have height and
#           weight (centres 1 and 5). Same ED/ES rule and error attribution as step 9c.
#   Part B. the locked 5-class ACDC rule applied to
#           the REFERENCE measurements of M&Ms-1 patients outside DCM/HCM/NOR
#           (ARV, IHD, HHD, ...). Secondary, descriptive analysis.
# =============================================================================
import os, io, re, glob, zipfile, numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import distance_transform_edt
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
S9, S14 = f'{DRIVE}/step9_mnms_external', f'{DRIVE}/step14_robustness'
OUT = f'{DRIVE}/step15_robustness'; os.makedirs(OUT, exist_ok=True)
LOCAL = '/content/mnms15'; os.makedirs(LOCAL, exist_ok=True)
SEED = 42
PRIMARY = ['DCM', 'HCM', 'NOR']
IDX = ['RV_ED', 'RV_ES', 'MYO_ED', 'MYO_ES', 'LV_ED', 'LV_ES', 'LV_mass']   # indexed to BSA
pd.set_option('display.width', 240)
bsa = lambda h_cm, w_kg: np.sqrt(h_cm * w_kg / 3600.0)

# ---------------- ACDC
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y = np.searchsorted(classes, folds.label.values)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
F = {c: j for j, c in enumerate(FEATS)}
A_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
A_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
hw = pd.read_csv(f'{S14}/acdc_height_weight.csv').set_index('patient').loc[order]
B_acdc = bsa(hw.height.values, hw.weight.values)
ii = [F[c] for c in IDX]

def index(X, b):
    X = X.copy(); X[..., ii] = X[..., ii] / b.reshape(-1, *([1] * (X.ndim - 1)))
    return X

def fit(X, yy):
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                        {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy').fit(X, yy).best_estimator_

head_raw = fit(A_gt, y)
head_idx = fit(index(A_gt, B_acdc), y)

# internal check on ACDC: same folds, predicted measurements, indexed vs not
pr_raw, pr_idx = np.zeros(len(y), int), np.zeros(len(y), int)
for k in sorted(folds.fold.unique()):
    tr, te = folds.fold.values != k, folds.fold.values == k
    pr_raw[te] = fit(A_gt[tr], y[tr]).predict(A_pr[te])
    pr_idx[te] = fit(index(A_gt[tr], B_acdc[tr]), y[tr]).predict(index(A_pr[te], B_acdc[te]))
print(f'ACDC out-of-fold balanced accuracy: non-indexed {balanced_accuracy_score(y, pr_raw):.2f} | '
      f'BSA-indexed {balanced_accuracy_score(y, pr_idx):.2f}')

# ---------------- M&Ms primary cohort (cached measurements) + ED/ES rule
R = pd.DataFrame(list(np.load(f'{S9}/mnms_concepts.npz', allow_pickle=True)['res']))
ZIP = glob.glob('/content/drive/MyDrive/**/M&M*.zip', recursive=True)[0]
z = zipfile.ZipFile(ZIP)
meta = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if 'MnMs_Dataset_information' in n][0])))
meta['External code'] = meta['External code'].astype(str).str.strip()
mi = meta.set_index('External code')
R['centre'] = R.code.map(mi['Centre'])
h = pd.to_numeric(R.code.map(mi['Height']), errors='coerce').values
w = pd.to_numeric(R.code.map(mi['Weight']), errors='coerce').values
if np.nanmedian(h) < 3: h = h * 100                                  # metres -> cm if needed
Cg, Cp = np.stack(list(R['gt'])), np.stack(list(R['pred']))

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
Cg[sw], Cp[sw] = swap_phases(Cg[sw]), swap_phases(Cp[sw])
pidx = np.searchsorted(classes, PRIMARY)
yv = np.searchsorted(PRIMARY, R.label.values)

def dec3(model, X):
    L = model.decision_function(X)[:, pidx]
    return L.argmax(1)

# =============================================================================
# Part A. BSA indexing
# =============================================================================
has = (h > 0) & (w > 0)
b = bsa(h[has], w[has])
print(f'\nPart A. M&Ms-1 primary patients with height and weight: {has.sum()}/{len(R)}')
print(pd.crosstab(R.label[has], R.centre[has], margins=True).to_string())
print(f'BSA: M&Ms-1 subset median {np.median(b):.2f} m2 | ACDC median {np.median(B_acdc):.2f} m2')

rows = []
for name, model, Xp, Xg in [('non-indexed (locked)', head_raw, Cp[has], Cg[has]),
                            ('BSA-indexed', head_idx, index(Cp[has], b), index(Cg[has], b))]:
    p, r_ = dec3(model, Xp), dec3(model, Xg)
    yy = yv[has]; wrong = p != yy
    typ = np.where(~wrong, 'CORRECT', np.where(r_ == yy, 'SEG', 'RULE'))
    rows.append(dict(rule=name, n=has.sum(), bal_acc=balanced_accuracy_score(yy, p),
                     bal_acc_refmeas=balanced_accuracy_score(yy, r_), errors=wrong.sum(),
                     SEG=(typ == 'SEG').sum(), RULE=(typ == 'RULE').sum(),
                     DCM_recall=np.mean(p[yy == 0] == 0), HCM_recall=np.mean(p[yy == 1] == 1), NOR_recall=np.mean(p[yy == 2] == 2)))
TA = pd.DataFrame(rows)
print(TA.round(2).to_string(index=False))
TA.to_csv(f'{OUT}/step15_bsa.csv', index=False)

# =============================================================================
# Part B. Patients outside DCM/HCM/NOR (reference measurements, 5-class rule)
# =============================================================================
def thickness_per_slice(myo, zooms):
    out = []
    for k in range(myo.shape[2]):
        m = myo[:, :, k]
        if m.sum() >= 10:
            out.append(2 * distance_transform_edt(m, sampling=zooms[:2]).max() - np.mean(zooms[:2]))
    return np.array(out) if out else np.array([0.0])

def rich_features(lab_ed, lab_es, z_ed, z_es):     # identical to step 3
    ved, ves = np.prod(z_ed) / 1000, np.prod(z_es) / 1000
    v = {f'{s}_{ph}': (lab == c).sum() * vv for ph, lab, vv in (('ED', lab_ed, ved), ('ES', lab_es, ves))
         for s, c in (('RV', 1), ('MYO', 2), ('LV', 3))}
    t_ed, t_es = thickness_per_slice(lab_ed == 2, z_ed), thickness_per_slice(lab_es == 2, z_es)
    eps = 1e-6
    return dict(**v,
                LV_EF=100 * (v['LV_ED'] - v['LV_ES']) / max(v['LV_ED'], eps),
                RV_EF=100 * (v['RV_ED'] - v['RV_ES']) / max(v['RV_ED'], eps),
                LV_mass=1.05 * v['MYO_ED'],
                RV_LV_ED=v['RV_ED'] / max(v['LV_ED'], eps), RV_LV_ES=v['RV_ES'] / max(v['LV_ES'], eps),
                MYO_LV_ED=v['MYO_ED'] / max(v['LV_ED'], eps), MYO_LV_ES=v['MYO_ES'] / max(v['LV_ES'], eps),
                WT_ED_max=t_ed.max(), WT_ED_mean=t_ed.mean(), WT_ED_std=t_ed.std(), WT_ED_min=t_ed.min(),
                WT_ES_max=t_es.max(), WT_ES_mean=t_es.mean(), WT_ES_std=t_es.std(), WT_ES_min=t_es.min())

REMAP = np.array([0, 3, 2, 1], np.uint8)          # M&Ms (LV=1, MYO=2, RV=3) -> ACDC (RV=1, MYO=2, LV=3)
gt_of = {re.sub(r'_sa_gt\.nii(\.gz)?$', '', os.path.basename(n)): n for n in z.namelist() if re.search(r'_sa_gt\.nii(\.gz)?$', n)}
O = meta[meta['External code'].isin(gt_of) & ~meta['Pathology'].isin(PRIMARY)].reset_index(drop=True)
print(f'\nPart B. M&Ms-1 patients outside DCM/HCM/NOR with contours: {len(O)}')
feats_o = []
for _, r in O.iterrows():
    fg = z.extract(gt_of[r['External code']], LOCAL)
    g = nib.load(fg); zooms = tuple(float(v) for v in g.header.get_zooms()[:3])
    G = np.rint(g.get_fdata()).astype(np.uint8)
    lab_t = [t for t in range(G.shape[3]) if G[..., t].max() > 0]
    a, c = REMAP[G[..., lab_t[0]]], REMAP[G[..., lab_t[-1]]]
    if (a == 3).sum() < (c == 3).sum(): a, c = c, a                 # ED = frame with the larger LV (same rule as 9c)
    ft = rich_features(a, c, zooms, zooms)
    feats_o.append([ft[k] for k in FEATS])
    os.remove(fg)
Xo = np.array(feats_o)
po = classes[head_raw.predict(Xo)]
print('Locked 5-class ACDC rule on reference measurements (rows = M&Ms-1 label, columns = ACDC class assigned):')
print(pd.crosstab(O['Pathology'].values, po, margins=True).to_string())
print('\nMedian reference measurements per M&Ms-1 label:')
print(pd.DataFrame(dict(label=O['Pathology'].values, LV_EF=Xo[:, F['LV_EF']], RV_EF=Xo[:, F['RV_EF']],
                        LV_EDV=Xo[:, F['LV_ED']], RV_EDV=Xo[:, F['RV_ED']], RV_LV_ED=Xo[:, F['RV_LV_ED']],
                        WT_ED_max=Xo[:, F['WT_ED_max']])).groupby('label').median().round(1).to_string())
pd.DataFrame(dict(code=O['External code'], centre=O['Centre'], label=O['Pathology'], acdc_class=po)).to_csv(
    f'{OUT}/step15_secondary.csv', index=False)
print('\nSaved to', OUT)
