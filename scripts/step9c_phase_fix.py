# =============================================================================
# STEP 9c — M&Ms with a physiological ED/ES rule (CPU, ~2 min, no re-segmentation)
#
# Data check (step 9b): in M&Ms the CSV ED/ES are 0/0 for ~half of Centre 4 and the
# larger LV is NOT at the CSV "ED" frame for most of Centre 5 (label numbering verified
# identical in all centres: label 1 is enclosed by myocardium = LV).
# Rule (by definition of end-diastole, label-free): of the two analysed frames,
#   ED = the one with the LARGER PREDICTED LV volume. Decided from the deterministic
#   prediction only (never from ground truth) and applied to gt, pred and all samples.
# Nothing else changes: same locked ACDC model, kappa, flag threshold, conformal qhat.
# Also reported (clearly labelled as sensitivity analysis, post hoc): without Centre 3,
# whose DCM labels have a median reference LV_EF > 60%.
# =============================================================================
import os, io, glob, zipfile, numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.stats import fisher_exact, beta, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, NotImplementedError):
    pass
DRIVE = os.environ.get('ACDC_DRIVE', '/content/drive/MyDrive/CMR_concept_bottleneck')
S8, OUT = f'{DRIVE}/step8_patient_uncertainty', f'{DRIVE}/step9_mnms_external'
SEED, S, FLAG, ALPHA, B = 42, 20, 0.20, 0.10, 2000
PRIMARY = ['DCM', 'HCM', 'NOR']
rng = np.random.default_rng(SEED)

# ---------------- locked ACDC model (identical to step 9)
folds = pd.read_csv(f'{DRIVE}/folds.csv')
order = folds.patient.values
classes = np.array(sorted(folds.label.unique()))
y_acdc = np.searchsorted(classes, folds.label.values)
rf = pd.read_csv(f'{DRIVE}/step3_challenge_baseline/rich_features.csv')
FEATS = [c for c in rf.columns if c not in ('patient', 'source')]
A_gt = rf[rf.source == 'gt'].set_index('patient').loc[order, FEATS].values
A_pr = rf[rf.source == 'pred'].set_index('patient').loc[order, FEATS].values
A_cs = np.load(f'{S8}/concept_samples.npz')['CS']
head = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight='balanced')),
                    {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='balanced_accuracy').fit(A_gt, y_acdc).best_estimator_
KAPPA = float(np.median(np.abs(A_pr - A_gt) / (A_cs.std(1) + 1e-9)) / 0.6745)
P_oof = np.load(f'{S8}/oof_probs_plain.npy')
cal = 1 - P_oof[np.arange(len(y_acdc)), y_acdc]
QHAT = np.quantile(cal, min(1, np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal)), method='higher')
pidx = np.searchsorted(classes, PRIMARY)

# ---------------- M&Ms cached measurements + centre
R = pd.DataFrame(list(np.load(f'{OUT}/mnms_concepts.npz', allow_pickle=True)['res']))
z = zipfile.ZipFile(glob.glob('/content/drive/MyDrive/**/M&M*.zip', recursive=True)[0])
meta = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if 'MnMs_Dataset_information' in n][0])))
meta['External code'] = meta['External code'].astype(str).str.strip()
R['centre'] = R.code.map(meta.set_index('External code')['Centre'])
n = len(R)
Cg, Cp, CS = np.stack(list(R['gt'])), np.stack(list(R['pred'])), np.stack(list(R['samples']))
F = {c: j for j, c in enumerate(FEATS)}

def swap_phases(X):
    """swap ED <-> ES in a (..., 21) feature array and recompute the derived features."""
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

sw = Cp[:, F['LV_ED']] < Cp[:, F['LV_ES']]                       # decided from the PREDICTION only
Cg[sw], Cp[sw], CS[sw] = swap_phases(Cg[sw]), swap_phases(Cp[sw]), swap_phases(CS[sw])
print(f'ED/ES swapped by the rule: {sw.sum()}/{n} patients')
print(pd.crosstab(R.centre, sw, rownames=['centre'], colnames=['swapped']).to_string())
print(f'negative reference LV_EF after the rule: {(Cg[:, F["LV_EF"]] < 0).sum()}')
print('\nMedian reference LV_EF by class and centre (sanity check):')
print(pd.DataFrame(dict(label=R.label, centre=R.centre, LV_EF=Cg[:, F['LV_EF']])).pivot_table(index='label', columns='centre', values='LV_EF', aggfunc='median').round(1).to_string())

# ---------------- locked evaluation (same as steps 9 / 9b)
def p3(X):
    L = head.decision_function(X)[:, pidx]
    return np.exp(L - logsumexp(L, 1, keepdims=True))
yv = np.searchsorted(PRIMARY, R.label.values); vend = R.vendor.values; cen = R.centre.values
dv = CS - CS.mean(1, keepdims=True)
Pp, Pg = p3(Cp), p3(Cg)
P5 = head.predict_proba(Cp)
pred, ref = Pp.argmax(1), Pg.argmax(1)
wrong, ref_ok = pred != yv, ref == yv
flips = np.zeros((n, len(FEATS)))
for t in range(n):
    for j in range(len(FEATS)):
        X = np.repeat(Cp[[t]], S, 0); X[:, j] += KAPPA * dv[t, :, j]
        flips[t, j] = np.mean(p3(X).argmax(1) != pred[t])
fmax = flips.max(1); flag = fmax >= FLAG; conf = 1 - Pp.max(1)
doubt = np.array(FEATS)[flips.argmax(1)]
cover = ((1 - P5) <= QHAT)[np.arange(n), pidx[yv]]
setsz = ((1 - P5) <= QHAT).sum(1)
typ = np.where(~wrong, 'CORRECT', np.where(ref_ok, 'SEG', 'RULE'))

def summary(m):
    yy, w = yv[m], wrong[m]
    return dict(n=m.sum(), bal_acc=balanced_accuracy_score(yy, pred[m]), bal_acc_refmeas=balanced_accuracy_score(yy, ref[m]),
                AUC=roc_auc_score(yy, Pp[m], multi_class='ovr', labels=range(3)) if len(np.unique(yy)) == 3 else np.nan,
                errors=w.sum(), SEG=(typ[m] == 'SEG').sum(), RULE=(typ[m] == 'RULE').sum(),
                flag_sens=flag[m][w].mean() if w.any() else np.nan, false_flag=flag[m][~w].mean(),
                AUROC_flip=roc_auc_score(w, fmax[m]) if 0 < w.sum() < m.sum() else np.nan,
                AUROC_conf=roc_auc_score(w, conf[m]) if 0 < w.sum() < m.sum() else np.nan,
                conformal_cov=cover[m].mean(), set_size=setsz[m].mean())
rows = [dict(subset='ALL', **summary(np.ones(n, bool)))]
rows += [dict(subset=v, **summary(vend == v)) for v in sorted(set(vend))]
rows += [dict(subset='ALL without centre 3 (sensitivity)', **summary(cen != 3))]
T = pd.DataFrame(rows)
pd.set_option('display.width', 240)
print('\nExternal results after the ED/ES rule (primary: DCM / HCM / NOR):')
print(T.round(3).to_string(index=False))
print('\nConfusion (rows = truth):', PRIMARY); print(confusion_matrix(yv, pred, labels=range(3)))

def cp(k, m):
    return (beta.ppf(0.025, k, m - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, m - k) if k < m else 1.0)
ok = typ == 'CORRECT'
print(f'\nFalse-flag rate on correct patients: {flag[ok].sum()}/{ok.sum()} CI {np.round(cp(flag[ok].sum(), ok.sum()), 2)}')
for tname in ('SEG', 'RULE'):
    m = typ == tname
    if m.sum() == 0: continue
    sel = m | ok
    odds, pv = fisher_exact([[flag[m].sum(), (~flag[m]).sum()], [flag[ok].sum(), (~flag[ok]).sum()]], alternative='greater')
    bs = []
    for _ in range(B):
        i = rng.choice(np.where(sel)[0], sel.sum()); ww = m[i]
        if 0 < ww.sum() < len(i): bs.append(roc_auc_score(ww, fmax[i]) - roc_auc_score(ww, conf[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'{tname:4s} errors n={m.sum():3d}: flag sens {flag[m].mean():.2f} CI {np.round(cp(flag[m].sum(), m.sum()), 2)} | '
          f'Fisher p = {pv:.2g}, OR = {odds:.1f} | AUROC flip {roc_auc_score(m[sel], fmax[sel]):.3f} vs conf '
          f'{roc_auc_score(m[sel], conf[sel]):.3f}, diff CI [{lo:+.3f}, {hi:+.3f}]')

print('\nSpearman: per-patient sample SD vs real |pred - gt|')
sd = CS.std(1)
for c in ('LV_EF', 'RV_EF', 'LV_ED', 'RV_ED', 'MYO_ED'):
    r_, p_ = spearmanr(sd[:, F[c]], np.abs(Cp[:, F[c]] - Cg[:, F[c]])); print(f'  {c:7s}: rho = {r_:+.2f}  p = {p_:.2g}')
print('\nMain source of doubt among SEG errors:'); print(pd.Series(doubt[typ == 'SEG']).value_counts().head(5).to_string())

T.to_csv(f'{OUT}/step9c_results.csv', index=False)
pd.DataFrame(dict(code=R.code, vendor=vend, centre=cen, label=R.label, swapped=sw, pred=np.array(PRIMARY)[pred],
                  ref_decision=np.array(PRIMARY)[ref], type=typ, flip_max=fmax, flag=flag, doubt=doubt,
                  in_set=cover)).to_csv(f'{OUT}/step9c_per_patient.csv', index=False)
print('\nSaved to', OUT)
